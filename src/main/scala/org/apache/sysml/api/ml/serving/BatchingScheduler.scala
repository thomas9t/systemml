package org.apache.sysml.api.ml.serving
import java.util.concurrent.ConcurrentHashMap
import scala.math.{floor, max}

trait BatchingScheduler extends Scheduler {

    val modelBatchSizes = new ConcurrentHashMap[String, ConcurrentHashMap[String,Int]]()

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
        val prevSize = modelBatchSizes.get(execType).get(model)
        val decreaseSize = if (prevSize > 10) max(floor(prevSize*0.90).toInt, 1) else max(prevSize-1, 1)
        modelBatchSizes.get(execType).put(model,
            if (latency < latencyObjective.toNanos) prevSize+1 else decreaseSize)
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
        var shortFuse = Set[String]()
        val keyIterator = modelQueues.keys()
        while (keyIterator.hasMoreElements) {
            val name = keyIterator.nextElement()
            val qsize = modelQueues.get(name).size()
            if (qsize > 0) {
                val nextRequest = modelQueues.get(name).peek()
                assert(nextRequest != null, "Something is wrong. Next request should not be null")

                if (checkShortFuse(nextRequest)) {
                    shortFuse += name
                }

                if (qsize >= getOptimalBatchSize(name, execType) || qsize == 1) {
                    batchableModels += name
                }
            }
        }

        if (shortFuse.nonEmpty) shortFuse else batchableModels
    }

    /**
      * Returns a boolean value indicating whether or not a request has exceeded its latency objective by more
      * than 20%
      */
    def checkShortFuse(request: SchedulingRequest) : Boolean = {
        val elapsed = System.nanoTime() - request.receivedTime
        val shortFuse = elapsed >= latencyObjectives.get(request.model.name).toNanos
        shortFuse
    }
}
