package org.apache.sysml.api.ml.serving

import java.lang.ref.WeakReference
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.LongAdder

import org.apache.sysml.api.jmlc.PreparedScript
import org.apache.sysml.hops.OptimizerUtils
import org.apache.sysml.api.jmlc.Connection
import org.apache.sysml.conf.ConfigurationManager
import org.apache.sysml.parser.Expression.ValueType
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject
import org.apache.sysml.runtime.matrix.data.{InputInfo, OutputInfo}
import org.apache.sysml.runtime.matrix.{MatrixCharacteristics, MetaDataFormat}

class WeightCache {
    val cache = new ConcurrentHashMap[String,WeakReference[MatrixObject]]()
    val liveWeights = new ConcurrentHashMap[String,MatrixObject]()
    val refCounts = new ConcurrentHashMap[String,LongAdder]()
    val cacheWeight = new LongAdder()
    val dummyObj = new LongAdder()
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
        println("ACQUIRING WEIGHT: " + path)
        // If the MO is live then return it immediately
        if (liveWeights.containsKey(path)) {
            println("WEIGHT IS LIVE - RETURN IMMEDIATE " + path)
            refCounts.get(path).increment()
            return liveWeights.get(path)
        }

        // otherwise check if it's still around as a soft reference
        // if so - promote it to a live object and return it
        // if it still exists, there must be enough memory to hold it
        // we need to synchronize though because we don't want this to
        // be deleted while we're accessing it
        if (cache.containsKey(path) && cache.get(path).get() != null) {
            cache.synchronized {
                if (cache.containsKey(path) && cache.get(path).get() != null) {
                    println("WEIGHT IS CACHED - RETURN FROM CACHE: " + path)
                    val mo = cache.get(path).get()
                    cacheWeight.add(mo.getDataSize)
                    liveWeights.put(path, mo)
                    refCounts.get(path).increment()
                    return mo
                }
            }
        }

        // Now if the object was not cached and was not live we need to actually get it from
        // disk
        if (!cache.containsKey(path) || cache.get(path).get() == null) {
            println("READING WEIGHT ALL THE WAY FROM DISK: " + path)
            dummyObj.synchronized {
                if (!cache.containsKey(path) || cache.get(path).get() == null) {
                    val mo = readWeight(path)
                    liveWeights.put(path, mo)
                    refCounts.putIfAbsent(path, new LongAdder())
                }
            }
        }

        val mo = liveWeights.get(path)
        refCounts.get(path).increment()
        mo
    }

    def release(path: String) : Unit = {
        println("RELEASING WEIGHT: " + path)
        refCounts.get(path).decrement()
        if (refCounts.get(path).longValue() == 0) {
            cache.synchronized {
                if (refCounts.get(path).longValue() == 0) {
                    println("TRANSFERRING WEIGHT TO WEAK REFERENCE: " + path)
                    val mo = liveWeights.remove(path)
                    cacheWeight.add(-1*mo.getDataSize)
                    cache.put(path, new WeakReference[MatrixObject](mo))
                }
            }
        }
    }

    def isCached(path: String) : Boolean = {
        liveWeights.containsKey(path) || (cache.containsKey(path) && cache.get(path).get() != null)
    }

}

class ModelCache(maxMemoryBytes: Long) {
    var models = new ConcurrentHashMap[String, Model]()
    var modelRefCounts = new ConcurrentHashMap[String, LongAdder]()
    val weightCache = new WeightCache()

    // TODO: Add logic to keep track of intermediates as well as live weights and block until enough space is available
    def acquire(name: String, execType: String) : PreparedScript = {
        println("ACQUIRING MODEL: " + name + " => " + modelRefCounts.get(name).longValue())
        // if the model has non-zero refcount then all weights are
        // guaranteed to be already pinned, so we can return immediately
        if (modelRefCounts.get(name).longValue() > 0) {
            println("RETURN IMMEDIATE: " + name)
            modelRefCounts.get(name).increment()
            return models.get(name).script(execType)
        }

        // otherwise we need to re-pin the weights, possibly reading them from disk
        // but this needs to be synchronized by model
        models.get(name).synchronized {
            if (modelRefCounts.get(name).longValue() == 0) {
                println("ACQUIRE WEIGHTS: " + name)
                val model = models.get(name)
                model.weightFiles.foreach(x => model.script(execType).setMatrix(x._1, weightCache.acquire(x._2), true))
                modelRefCounts.get(name).increment()
            } else {
                modelRefCounts.get(name).increment()
            }
        }

        println("DONE ACQUIRE MODEL: " + name)
        models.get(name).script(execType)
    }

    def release(name: String) : Unit = {
        println("RELEASING MODEL: " + name)
        modelRefCounts.get(name).decrement()
        if (modelRefCounts.get(name).longValue() == 0) {
            models.get(name).synchronized {
                if (modelRefCounts.get(name).longValue() == 0) {
                    println("RELEASING WEIGHTS FOR: " + name)
                    models.get(name).weightFiles.foreach { x => weightCache.release(x._2) }
                }
            }
        }
        println("DONE RELEASE MODEL: " + name)
    }

    def put(model: Model) : Unit = {
        models.putIfAbsent(model.name, model)
        modelRefCounts.putIfAbsent(model.name, new LongAdder())
    }
}
