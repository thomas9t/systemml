package org.apache.sysml.api.ml.serving

import java.util
import java.util.concurrent.atomic.LongAdder

import org.apache.sysml.api.jmlc.{Connection, PreparedScript}
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.utils.PersistentLRUCache

trait ModelManager {

    val conn: Connection = new Connection()

    // for GPU or mixed executors we need to have multiple prepared scripts
    var scripts: Map[String, util.HashMap[JmlcExecutor, PreparedScript]] = Map()

    // we can have a single prepared script for CPU only executors
    var cpuScripts: Map[String, PreparedScript] = Map()

    val availableMemory = new LongAdder

    var totalMemory = 0L

    var cleanupEnabled = true

    var memCheckEnabled = true

    var models: Map[String, Model] = Map()

    def setAvailableMemory(memBytes: Long) : Unit = {
        println("SETTING TOTAL MEMORY AVAILABLE TO: " + memBytes)
        totalMemory = memBytes
        availableMemory.reset()
        availableMemory.add(memBytes)
    }

    def getAvailableMemory : Long = { availableMemory.longValue() }

    def acquireMemory(bytes: Long) : Long = {
        // if memory checking is not enabled just always say they get the memory
        if (!memCheckEnabled || bytes == 0)
            return bytes

        // otherwise check to see if there is enough memory to meet the request
        if (bytes <= availableMemory.longValue()) {
            this.synchronized {
                if (bytes <= availableMemory.longValue()) {
                    println("GRANTED: " + bytes + "/" + availableMemory)
                    availableMemory.add(-1 * bytes)
                    return bytes
                }
            }
        }
        // not enough memory available :(
        -1
    }

    def releaseMemory(bytes: Long) : Unit = {
        if (bytes > 0) {
            println("RELEASING: " + bytes)
            availableMemory.add(bytes)
            println("MEMORY IS NOW: " + availableMemory.longValue())
        }
    }

    def getPreparedScript(name: String, executor: JmlcExecutor) : PreparedScript = {
        if (scripts(name).containsKey(executor))
            return scripts(name).get(executor)

        // otherwise we may need to compile...
        // Note: this logic will need to be revisited if we add more executor types beyond CPU and GPU
        val model = models(name)
        val inputs = model.weightFiles.keys.toArray[String] ++ Array[String](model.inputVarName)
        if (executor.getExecType == "CPU") {
            if (!cpuScripts.contains(name))
                cpuScripts += name -> conn.prepareScript(model.dml, inputs, Array[String](model.outputVarName))
            else
                scripts(name).put(executor, cpuScripts(name))
        } else if (executor.getExecType == "GPU") {
            val script = conn.prepareScript(model.dml, inputs, Array[String](model.outputVarName),
                true, true, executor.getGpuIndex)
            scripts(name).put(executor, script)
        }
        scripts(name).get(executor)
    }

    def clearPinnedData(name: String) : Unit = {
        val scriptIterator = scripts(name).values().iterator()
        while (scriptIterator.hasNext) {
            val script = scriptIterator.next()
            script.clearPinnedData()
        }

    }

    def disableCleanup() : Unit = { cleanupEnabled = false }

    def disableMemcheck() : Unit = { memCheckEnabled = false }

    def put(model: Model): Unit

    def get(name: String): Model

    def putWeight(name: String, weight: MatrixBlock) : Unit

    def acquire(name: String, executor: JmlcExecutor) : PreparedScript

    def release(name: String) : Unit
}

object ReferenceCountedModelManager extends ModelManager {
    var modelRefCounts: Map[String,LongAdder] = Map()
    var weightCache : PersistentLRUCache = _

    override def setAvailableMemory(maxBytes: Long) : Unit = {
        super.setAvailableMemory(maxBytes)
        weightCache = new PersistentLRUCache((0.80*maxBytes).toLong)
        weightCache.enableReadOnlyMode(true)
    }

    def tryAllocMem(name: String, batchSize: Int) : Long = {
        // TODO: More sophisticated memory management
        val extraMem = (0.5*models(name).weightMem).toLong
        val weightMem = if (modelRefCounts(name).longValue() > 0) 0L else models(name).weightMem
        val memReceived = acquireMemory(extraMem + weightMem)
        if (memReceived < 0) memReceived else extraMem
    }

    def isCached(name: String) : Boolean = { modelRefCounts(name).longValue() > 0 }

    def acquire(name: String, executor: JmlcExecutor) : PreparedScript = {
         println("ACQUIRING MODEL: " + name + " => " + modelRefCounts(name).longValue())
        // the "has pinned vars" check is necessary because there may be separate prepared scripts
        // compiled for various executor types (e.g. GPU/CPU)
        // TODO: Better handling of the GPU pinned variables issue

        // check if this model has been compiled for this executor

        val ps = getPreparedScript(name, executor).clone(false)
        /*if (modelRefCounts(name).longValue() > 0 && ps.hasPinnedVars) {
            modelRefCounts(name).increment()
            return ps.clone(false)
        }*/

        // otherwise we need to re-pin the weights, possibly reading them from disk
        val model = models(name)
        model.synchronized {
            if (modelRefCounts(name).longValue() == 0)
                model.weightFiles.foreach(x => ps.setMatrix(x._1, weightCache.getAsMatrixBlock(x._2), true))
            modelRefCounts(name).increment()
        }
        println("DONE ACQUIRING MODEL: " + name)
        ps
    }

    override def disableCleanup(): Unit = {
        super.disableCleanup()
        println("CLEANUP IS DISABLED")
    }

    def release(name: String) : Unit = {
        modelRefCounts(name).decrement()

        println("RELEASE MODEL: " + name + " => " + modelRefCounts(name).longValue())
        if (modelRefCounts(name).longValue() == 0 && cleanupEnabled) {
            models(name).synchronized {
                if (modelRefCounts(name).longValue() == 0) {
                    println("ACTUALLY RELEASING THE MODEL")
                    clearPinnedData(name)
                    println("CALLING RELEASE MEMORY")
                    releaseMemory(models(name).weightMem)
                }
            }
        }
    }

    def put(model: Model) : Unit = {
        models += (model.name -> model)
        modelRefCounts += (model.name -> new LongAdder())
        scripts += (model.name -> new util.HashMap[JmlcExecutor, PreparedScript]())
    }

    def putWeight(name: String, weight: MatrixBlock) : Unit = {
        weightCache.put(name, weight)
    }

    def get(name: String) : Model = { models(name) }

}
