package org.apache.sysml.api.ml.serving

import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.LongAdder

import scala.concurrent.Future
import scala.math.min
import scala.concurrent.duration.Duration

class ExecutorQueueMananger(scheduler: BatchingScheduler) extends Runnable {
    var _shutDown = false

    def shutdown(): Unit = { _shutDown = true }

    override def run() : Unit = {
        println("Hi From Executor Queue Manager!")
        while (!_shutDown) {
            scheduler.dummyResponse.synchronized {
                val schedulableModels = scheduler.executorTypes.map(
                    x => scheduler.getSchedulableModels(x)).reduce(_ union _)
                if (schedulableModels.nonEmpty) {
                    for (m <- schedulableModels) {
                        val queue = getLowestUtilizationQueue(m)
                        val nextBatchSize = min(scheduler.modelQueues.get(m).size(),
                            scheduler.getOptimalBatchSize(m, queue.getExecType))
                        val nextRequest = scheduler.modelQueues.get(m).peek()
                        assert(nextBatchSize > 0, "Something is wrong - batch size should not be zero")
                        val nextBatch = Batch(
                            nextBatchSize, scheduler.getExpectedExecutionTime(m, nextBatchSize, queue.getExecType),
                            nextRequest.receivedTime - System.nanoTime(), nextRequest.model.name)
                        queue.enqueue(nextBatch)
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
    val availableCpuMemory = new LongAdder()

    override def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        super.start(numCores, cpuMemoryBudgetInBytes, gpus)
        //availableCpuMemory.add((cpuMemoryBudgetInBytes*0.90).toLong)

        availableCpuMemory.add(0)
        queueManager = new Thread(new ExecutorQueueMananger(this))
        queueManager.start()
    }

    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        val localQueue = executorQueues.get(executor)

        if (localQueue.size() > 0 || globalSchedulingQueues.get(executor.getExecType).size() > 0) {
            dummyResponse.synchronized {
                if (localQueue.size() > 0 || globalSchedulingQueues.get(executor.getExecType).size() > 0) {
                    val localExecTime = localQueue.getExpectedExecutionTime
                    val globalExecTime = globalSchedulingQueues.get(executor.getExecType).getExpectedExecutionTime
                    val batch = if (localExecTime >= globalExecTime)
                        localQueue.dequeue() else  globalSchedulingQueues.get(executor.getExecType).dequeue()
                    val model = modelManager.get(batch.modelName)
                    val numToDequeue = getLargestPossibleBatchSize(
                        min(batch.size, modelQueues.get(batch.modelName).size()), model.memoryEstimator)
                    if (numToDequeue > 0) {
                        for (_ <- 0 until numToDequeue) {
                            ret :+= modelQueues.get(batch.modelName).poll()
                            assert(ret != null, "Something is wrong - request should not be null!")
                        }

                        if (executor.prevModel.nonEmpty) {
                            unsetModelLocality(batch.modelName, localQueue)
                        }
                        setModelLocality(batch.modelName, localQueue)
                    }
                }
            }
        }
        ret
    }

    /**
      * Returns the largest possible batch size which can be executed given the currently available
      * memory
      * @param b desired batch size
      * @param memUseEstimator function which estimates memory use given a batch size
      * @return the largest batch size less than or equal to b which does not exceed available memory
      */
    def getLargestPossibleBatchSize(b: Int, memUseEstimator: Int => Long): Int = {
        val lb = memUseEstimator(b)
        val ub = memUseEstimator(b+1)
        val limit = availableCpuMemory.longValue()
        if (ub < limit)
            return getLargestPossibleBatchSize(b*2, memUseEstimator)
        if (lb > limit)
            return getLargestPossibleBatchSize(b/2, memUseEstimator)
        b
    }

    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which prediction
      * @return
      */
    override private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val statistics = if (_statistics) RequestStatistics() else null
        val schedulingRequest = SchedulingRequest(
            request, model, new CountDownLatch(1), System.nanoTime(), null, statistics)
        statistics.queueSize = modelQueues.get(model.name).size
        requestQueue.add(schedulingRequest)
        modelQueues.get(model.name).add(schedulingRequest)

        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case _ : scala.concurrent.TimeoutException => dummyResponse
        }
    }
}
