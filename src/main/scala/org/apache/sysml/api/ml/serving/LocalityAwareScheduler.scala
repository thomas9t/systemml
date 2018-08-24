package org.apache.sysml.api.ml.serving

import java.util.concurrent.CountDownLatch

import scala.concurrent.Future
import scala.math.min
import scala.concurrent.duration.Duration

class ExecutorQueueMananger(scheduler: BatchingScheduler) extends Runnable {
    var _shutDown = false

    def shutdown(): Unit = { _shutDown = true }

    override def run() : Unit = {
        println("Hi From Executor Queue Manager!")
        while (!_shutDown) {
            scheduler.synchronized {
                val schedulableModels = scheduler.executorTypes.map(
                    x => scheduler.getSchedulableModels(x)).reduce(_ union _)
                if (schedulableModels.nonEmpty) {
                    for (m <- schedulableModels) {
                        println("BEGIN ROUTING")
                        val queue = getLowestUtilizationQueue(m)
                        var requests = Array[SchedulingRequest]()
                        val nextBatchSize = min(scheduler.modelQueues.get(m).size(),
                            scheduler.getOptimalBatchSize(m, queue.getExecType))
                        for (_ <- 0 until nextBatchSize) {
                            val nextRequest = scheduler.modelQueues.get(m).poll()
                            requests :+= nextRequest
                        }
                        assert(requests.nonEmpty, "Something is Wrong - Requests should not be empty")
                        val nextBatch = Batch(
                            requests, scheduler.getExpectedExecutionTime(m, requests.length, queue.getExecType))
                        queue.enqueue(nextBatch)
                        println("DONE ROUTING")
                    }
                }
            }
        }
    }

    def getLowestUtilizationQueue(model: String): BatchQueue = {
        var queues = Array[BatchQueue]()
        val iterator = scheduler.modelLocality.get(model).keys()
        while (iterator.hasMoreElements) { queues :+= iterator.nextElement() }
        queues.minBy(_.getExpectedExecutionTime)
    }
}

class LocalityAwareScheduler(override val timeout: Duration) extends BatchingScheduler {
    var queueManager : Thread = _

    override def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        super.start(numCores, cpuMemoryBudgetInBytes, gpus)
        queueManager = new Thread(new ExecutorQueueMananger(this))
        queueManager.start()
    }

    override def schedule(executor: JmlcExecutor) : Batch = {
        var batch = Batch(Array[SchedulingRequest](), -1)
        val localQueue = executorQueues.get(executor)
        if (localQueue.size() > 0 || globalSchedulingQueues.get(executor.getExecType).size() > 0) {
            println("TRYING TO SCHEDULE")
            dummyResponse.synchronized {
                println("BEGIN SCHEDULE ACTUAL")
                if (localQueue.size() > 0 || globalSchedulingQueues.get(executor.getExecType).size() > 0) {
                    val localExecTime = localQueue.getExpectedExecutionTime
                    val globalExecTime = globalSchedulingQueues.get(executor.getExecType).getExpectedExecutionTime
                    if (localExecTime >= globalExecTime) {
                        batch = localQueue.dequeue()
                    } else {
                        batch = globalSchedulingQueues.get(executor.getExecType).dequeue()
                    }
                    assert(batch.requests.nonEmpty, "Something is wrong - got empty batch")
                    if (executor.prevModel.nonEmpty) {
                        unsetModelLocality(batch.requests.last.model.name, localQueue)
                    }
                    setModelLocality(batch.requests.last.model.name, localQueue)
                }
                println("DONE SCHEDULE ACTUAL")
            }
        }
        batch
    }

    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which prediction
      * @return
      */
    override private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        println("ENQUEUEING REQUEST FOR: " + model.name)
        val statistics = if (_statistics) RequestStatistics() else null
        val schedulingRequest = SchedulingRequest(
            request, model, new CountDownLatch(1), System.nanoTime(), null, statistics)
        statistics.queueSize = modelQueues.get(model.name).size
        requestQueue.add(schedulingRequest)
        modelQueues.get(model.name).add(schedulingRequest)
        println("DONE ENQUEUING")

        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e : scala.concurrent.TimeoutException => dummyResponse
        }
    }
}
