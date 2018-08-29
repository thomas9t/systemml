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
    var models: Map[String, Model] = Map()

    def put(model: Model): Unit

    def get(name: String): Model

    def acquire(name: String, execType: String) : PreparedScript

    def release(name: String) : Unit
}

object BasicModelManager extends ModelManager {

    def put(model: Model): Unit = { models += (model.name -> model) }

    def get(name: String): Model = { models(name) }

    def acquire(name: String, execType: String): PreparedScript = { get(name).script(execType) }

    def release(name: String): Unit = {}

}

object WeightCache {
    val cache = new ConcurrentHashMap[String,WeakReference[MatrixObject]]()
    val liveWeights = new ConcurrentHashMap[String,MatrixObject]()
    var refCounts: Map[String,LongAdder] = Map()
    val cacheWeight = new LongAdder()
    val conn = new Connection()

    def readWeight(path: String): MatrixObject = {
        println("READING WEIGHT: " + path + " FROM DISK")
        val mat = conn.readMatrix(path)
        val blocksize = ConfigurationManager.getBlocksize
        val mc = new MatrixCharacteristics(mat.getNumRows, mat.getNumColumns, blocksize, blocksize)
        val meta = new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo)
        val mo = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName, meta)
        mo.acquireModify(mat)
        mo.release()
        println("DONE READ WEIGHT: " + path)
        mo
    }

    def acquire(path: String) : MatrixObject = {
        println("ACQUIRING WEIGHT: " + path + " CACHE SIZE => " + cacheWeight.longValue())
        // If the MO is live then return it immediately
        if (liveWeights.containsKey(path)) {
            println("WEIGHT IS LIVE - RETURN IMMEDIATE " + path)
            refCounts(path).increment()
            return liveWeights.get(path)
        }

        // otherwise check if it's still around as a soft reference
        // if so - promote it to a live object and return it
        // if it still exists, there must be enough memory to hold it
        // we need to synchronize though because we don't want this to
        // be deleted while we're accessing it
        if (cache.containsKey(path)) {
            val mo = cache.get(path).get()
            if (mo != null) {
                println("WEIGHT IS CACHED - RETURN FROM CACHE: " + path)
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
        refCounts(path).increment()
        mo
    }

    def release(path: String) : Unit = {
        println("RELEASING WEIGHT: " + path)
        refCounts(path).decrement()
        if (refCounts(path).longValue() == 0) {
            println("TRANSFERRING WEIGHT TO WEAK REFERENCE: " + path)
            val mo = liveWeights.remove(path)
            if (mo != null) {
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
        println("ACQUIRING MODEL: " + name + " => " + modelRefCounts(name).longValue())
        // if the model has non-zero refcount then all weights are
        // guaranteed to be already pinned, so we can return immediately
        if (modelRefCounts(name).longValue() > 0) {
            println("RETURN IMMEDIATE: " + name)
            modelRefCounts(name).increment()
            return models(name).script(execType)
        }

        // otherwise we need to re-pin the weights, possibly reading them from disk
        println("ACQUIRE WEIGHTS: " + name)
        val model = models(name)
        model.weightFiles.foreach(x => model.script(execType).setMatrix(x._1, weightCache.acquire(x._2), true))
        modelRefCounts(name).increment()
        println("DONE ACQUIRE MODEL: " + name)

        models(name).script(execType)
    }

    def release(name: String) : Unit = {
        println("RELEASING MODEL: " + name)
        modelRefCounts(name).decrement()
        if (modelRefCounts(name).longValue() == 0) {
            println("RELEASING WEIGHTS FOR: " + name)
            models(name).weightFiles.foreach { x => weightCache.release(x._2) }
        }
        println("DONE RELEASE MODEL: " + name)
    }

    def put(model: Model) : Unit = {
        models += (model.name -> model)
        modelRefCounts += (model.name -> new LongAdder())
    }

    def get(name: String) : Model = { models(name) }
}
