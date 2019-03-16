package org.apache.sysml.api.ml.serving
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.LongAdder

import scala.math.{floor, max}

trait BatchingScheduler extends Scheduler {

    val modelBatchSizes = new ConcurrentHashMap[String, ConcurrentHashMap[String,Int]]()
    val expectedExecutionTimes = new ConcurrentHashMap[String, (LongAdder, LongAdder)]()

    def getOptimalBatchSize(model : String, execType: String) : Int = {
        modelBatchSizes.putIfAbsent(execType, new ConcurrentHashMap[String,Int]())
        modelBatchSizes.get(execType).putIfAbsent(model, 2)
        modelBatchSizes.get(execType).get(model)
    }

    override def onCompleteCallback(model: String,
                                    latency: Double,
                                    batchSize: Int,
                                    execType: String,
                                    execTime: Long): Unit = {
        if (batchSize > 1) {
            val latencyObjective = latencyObjectives.get(model)
            val prevSize = modelBatchSizes.get(execType).get(model)
            val decreaseSize = if (prevSize > 10) max(floor(prevSize * 0.90).toInt, 1) else prevSize - 1
            modelBatchSizes.get(execType).put(model,
                if (latency < latencyObjective.toNanos) prevSize + 1 else decreaseSize)

            // update expected execution times. For now we just assume this is a simple average
            val execTimeData = expectedExecutionTimes.get(model)
            execTimeData._1.add(execTime / batchSize)
            execTimeData._2.increment()
        }
    }

    def getExpectedExecutionTime(model: String) : Long = {
        expectedExecutionTimes.putIfAbsent(model, (new LongAdder(), new LongAdder()))
        val execTime = expectedExecutionTimes.get(model)
        val totalNumRequests = execTime._2.longValue()
        if  (totalNumRequests > 0) execTime._1.longValue() / execTime._2.longValue() else 0
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

                if (checkShortFuse(nextRequest, qsize)) {
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
      * Returns a boolean value if it would violate the latency threshold to execute the current number of models
      */
    def checkShortFuse(request: SchedulingRequest, numRequests: Int) : Boolean = {
        val elapsed = System.nanoTime() - request.receivedTime
        (elapsed + 1.1*numRequests*getExpectedExecutionTime(request.model.name)) > request.model.latencyObjective.toNanos
    }
}
