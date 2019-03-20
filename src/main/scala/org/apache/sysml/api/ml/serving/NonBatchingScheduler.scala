package org.apache.sysml.api.ml.serving

import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.LongAdder

import scala.concurrent.Future
import scala.concurrent.duration.Duration

object NonBatchingScheduler extends Scheduler {

    override def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        LOG.info(s"Starting Non Batching Scheduler with: ${numCores} CPUs and ${gpus} GPUs")
        super.start(numCores, cpuMemoryBudgetInBytes, gpus)
    }

    override def schedule(executor: JmlcExecutor): Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        dummyResponse.synchronized {
            if (requestQueue.size() > 0) {
                val request = requestQueue.poll()
                ret :+= request
            }
        }
        ret
    }

    var requestNum = new LongAdder
    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which prediction is desired
      * @return
      */
    override private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val statistics = if (_statistics) RequestStatistics() else null
        val schedulingRequest = SchedulingRequest(
            request, model, new CountDownLatch(1), System.nanoTime(), null, statistics)
        if (_statistics) statistics.queueSize = requestQueue.size()
        requestQueue.add(schedulingRequest)
        counter += 1
        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e : scala.concurrent.TimeoutException => dummyResponse
        }
    }

    override def onCompleteCallback(model: String, latency: Double, batchSize: Int, execType: String, execTime: Long): Unit = {}
}
