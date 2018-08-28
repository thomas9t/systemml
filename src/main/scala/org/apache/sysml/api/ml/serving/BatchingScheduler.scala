package org.apache.sysml.api.ml.serving

import java.util.concurrent.ConcurrentHashMap

import scala.math.floor

trait BatchingScheduler extends Scheduler {

    val modelBatchSizes = new ConcurrentHashMap[String, ConcurrentHashMap[String,Int]]()
    val execTimeEstimators = new ConcurrentHashMap[String, ConcurrentHashMap[String,RLSEstimator]]()

    def getOptimalBatchSize(model : String, execType: String) : Int = {
        modelBatchSizes.putIfAbsent(execType, new ConcurrentHashMap[String,Int]())
        modelBatchSizes.get(execType).putIfAbsent(model, 1)
        modelBatchSizes.get(execType).get(model)
    }

    override def onCompleteCallback(model: String,
                                    latency: Double,
                                    batchSize: Int,
                                    execType: String): Unit = {
        val latencyObjective = latencyObjectives.get(model)
        if (latency != latencyObjective.toNanos) {
            modelBatchSizes.synchronized({
                val prevSize = modelBatchSizes.get(execType).get(model)
                modelBatchSizes.get(execType).put(model,
                    if (latency < latencyObjective.toNanos) prevSize+2 else floor(prevSize*0.90).toInt)
            })
        }
//        execTimeEstimators.putIfAbsent(model, new ConcurrentHashMap[String,RLSEstimator]())
//        execTimeEstimators.get(model).putIfAbsent(execType, new RLSEstimator)
//        execTimeEstimators.get(model).get(execType).enqueueExample(batchSize, latency)
    }

    def getExpectedExecutionTime(model: String, batchSize: Int, execType: String) : Long = {
        latencyObjectives.get(model).toNanos
    }

    /**
      * Gets a list of models that are eligible to be run. A model is eligible to be run if it
      * has a greater number of requests enqueued than its optimal batch size.
      * @return A list of models which may be scheduled
      */
    def getSchedulableModels(execType: String) : Set[String] = {
        var batchableModels = Set[String]()
        var shortFuse = Set[(String,Long)]()
        val keyIterator = modelQueues.keys()
        while (keyIterator.hasMoreElements) {
            val name = keyIterator.nextElement()
            if (modelQueues.get(name).size() > 0) {
                val nextRequest = modelQueues.get(name).peek()
                if (checkShortFuse(nextRequest)) {
                    shortFuse += ((name, nextRequest.receivedTime - System.nanoTime()))
                }
                if (modelQueues.get(name).size() >= getOptimalBatchSize(name, execType)) {
                    batchableModels += name
                }
            }
        }

        var shortFuseArray = Set[String]()
        if (shortFuse.nonEmpty) {
            shortFuseArray += shortFuse.minBy(x => x._2)._1
        }

        if (shortFuseArray.nonEmpty) shortFuseArray else batchableModels
    }

    /**
      * Returns a boolean value indicating whether or not a request has exceeded its latency objective by more
      * than 20%
      */
    def checkShortFuse(request: SchedulingRequest) : Boolean = {
        val elapsed = System.nanoTime() - request.receivedTime
        val shortFuse = elapsed >= (1.2*latencyObjectives.get(request.model.name).toNanos).toLong
        shortFuse
    }
}
