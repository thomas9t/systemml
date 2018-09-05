package org.apache.sysml.api.ml.serving

import java.lang.ref.WeakReference
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.LongAdder

import org.apache.sysml.api.jmlc.PreparedScript
import org.apache.sysml.hops.OptimizerUtils
import org.apache.sysml.conf.ConfigurationManager
import org.apache.sysml.parser.Expression.ValueType
import org.apache.sysml.api.jmlc.Connection
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject
import org.apache.sysml.runtime.matrix.data.{InputInfo, OutputInfo}
import org.apache.sysml.runtime.matrix.{MatrixCharacteristics, MetaDataFormat}

trait ModelManager {

    var cleanupEnabled = true

    var models: Map[String, Model] = Map()

    def disableCleanup() : Unit = { cleanupEnabled = false }

    def put(model: Model): Unit

    def get(name: String): Model

    def acquire(name: String, execType: String) : PreparedScript

    def release(name: String) : Unit
}

object WeightCache {
    val cache = new ConcurrentHashMap[String,WeakReference[MatrixObject]]()
    val liveWeights = new ConcurrentHashMap[String,MatrixObject]()
    var refCounts: Map[String,LongAdder] = Map()
    val cacheWeight = new LongAdder()
    val conn = new Connection()
    var cleanupEnabled = true

    def readWeight(path: String): MatrixObject = {
        val mat = conn.readMatrix(path)
        val blocksize = ConfigurationManager.getBlocksize
        val mc = new MatrixCharacteristics(mat.getNumRows, mat.getNumColumns, blocksize, blocksize)
        val meta = new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo)
        val mo = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName, meta)
        mo.acquireModify(mat)
        mo.release()
        mo
    }

    def disableCleanup() : Unit = { cleanupEnabled = false }

    def acquire(path: String) : MatrixObject = {
        // If the MO is live then return it immediately
        val mat = liveWeights.getOrDefault(path, null)
        if (mat != null) {
            refCounts(path).increment()
            return mat
        }

        // otherwise check if it's still around as a soft reference
        // if so - promote it to a live object and return it
        // if it still exists, there must be enough memory to hold it
        // we need to synchronize though because we don't want this to
        // be deleted while we're accessing it
        if (cache.containsKey(path)) {
            val mo = cache.get(path).get()
            if (mo != null) {
                cacheWeight.add(mo.getDataSize)
                liveWeights.put(path, mo)
                refCounts(path).increment()
                return mo
            }
        }

        // Now if the object was not cached and was not live we need to actually get it from
        // disk possibly waiting until the cache has enough space to read it
        if (!cache.containsKey(path) || cache.get(path).get() == null) {
            this.synchronized {
                if (!cache.containsKey(path) || cache.get(path).get() == null) {
                    val mo = readWeight(path)
                    liveWeights.put(path, mo)
                    refCounts += (path -> new LongAdder())
                }
            }
        }

        val mo = liveWeights.get(path)
        cacheWeight.add(mo.getDataSize)
        refCounts(path).increment()
        mo
    }

    def release(path: String) : Unit = {
        refCounts(path).decrement()
        if (refCounts(path).longValue() == 0) {
            val mo = liveWeights.remove(path)
            if (mo != null && cleanupEnabled) {
                cacheWeight.add(-1 * mo.getDataSize)
                cache.put(path, new WeakReference[MatrixObject](mo))
            }
        }
    }

    def isCached(path: String) : Boolean = {
        liveWeights.containsKey(path) || (cache.containsKey(path) && cache.get(path).get() != null)
    }

}

object ReferenceCountedModelManager extends ModelManager {
    var modelRefCounts: Map[String,LongAdder] = Map()
    val weightCache = WeightCache

    // TODO: Add logic to keep track of intermediates as well as live weights and block until enough space is available
    def acquire(name: String, execType: String) : PreparedScript = {
        // println("ACQUIRING MODEL: " + name + " => " + modelRefCounts(name).longValue())
        // if the model has non-zero refcount then all weights are
        // guaranteed to be already pinned, so we can return immediately
        modelRefCounts(name).increment()
        if (modelRefCounts(name).longValue() > 1)
                return models(name).script(execType)

        // otherwise we need to re-pin the weights, possibly reading them from disk
        val model = models(name)
        model.weightFiles.foreach(x => model.script(execType).setMatrix(x._1, weightCache.acquire(x._2), true))
        modelRefCounts(name).increment()

        models(name).script(execType)
    }

    override def disableCleanup(): Unit = {
        super.disableCleanup()
        weightCache.disableCleanup()
        println("CLEANUP IS DISABLED")
    }

    def release(name: String) : Unit = {
        modelRefCounts(name).decrement()
        if (modelRefCounts(name).longValue() == 0 && cleanupEnabled) {
            models(name).script.foreach { x => x._2.clearParameters() }
            models(name).weightFiles.foreach { x => weightCache.release(x._2) }
        }
    }

    def put(model: Model) : Unit = {
        models += (model.name -> model)
        modelRefCounts += (model.name -> new LongAdder())
    }

    def get(name: String) : Model = { models(name) }
}
