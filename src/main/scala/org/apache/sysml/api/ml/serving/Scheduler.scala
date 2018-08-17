/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.api.ml.serving

import scala.concurrent.Future
import scala.concurrent.duration._
import java.util.concurrent._

import org.apache.sysml.runtime.instructions.gpu.context.{GPUContext, GPUContextPool}

import scala.concurrent.ExecutionContext
import scala.math.{min, floor}

case class SchedulingRequest(request: PredictionRequest,
                             model: Model,
                             latch: CountDownLatch,
                             receivedTime: Long,
                             var response: PredictionResponse = null,
                             statistics: RequestStatistics = null)

class JmlcExecutor(scheduler: Scheduler, execType: String, gCtx: GPUContext) extends Runnable {
    @volatile protected var _shouldShutdown: Boolean = false

    def shutdown(): Unit = {
        _shouldShutdown = true
    }

    def getExecType(): String = { execType }

    def run(): Unit = {
        while (!_shouldShutdown) {
            val requests = scheduler.schedule(this)
            if (requests.nonEmpty) {
                val responses = execute(requests)
                for ((req, resp) <- requests zip responses) {
                    req.response = resp
                    req.latch.countDown()
                }
            }
        }
    }

    def execute(requests: Array[SchedulingRequest]): Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.nonEmpty) {
            val start = System.nanoTime()
            val batchedMatrixData = BatchingUtils.batchRequests(requests)
            val batchingTime = System.nanoTime() - start
            val req = requests(0)
            val script = req.model.script(execType).clone(false)
            script.setGpuContext(gCtx)
            script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
            val execStart = System.nanoTime()
            val res = script.executeScript().getMatrixBlock(req.model.outputVarName)
            val execTime = System.nanoTime() - execStart
            responses = BatchingUtils.unbatchRequests(requests, res)
            val stop = System.nanoTime()
            scheduler.onCompleteCallback(req.model.name, stop - req.receivedTime, requests.length, execType)
            if (req.statistics != null)
                setStatistics(requests, start, batchingTime, execTime)
        }
        responses
    }

    def setStatistics(requests: Array[SchedulingRequest],
                      processingStartTime: Long, batchingTime: Long, execTime: Long): Unit = {
        for (req <- requests) {
            req.statistics.batchingTime = batchingTime
            req.statistics.execType = getExecType()
            req.statistics.batchSize = requests.length
            req.statistics.queueWaitTime = processingStartTime - req.receivedTime
        }
    }
}

trait Scheduler {
    var executorService: ExecutorService = _
    private var _statistics = true
    implicit val ec : ExecutionContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(10000))

    def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        val gCtxs = if (gpus != null) GPUContextPool.reserveAllGPUContexts() else null
        val numGpus = if (gCtxs != null) gCtxs.size() else 0
        executorService = Executors.newFixedThreadPool(numCores + numGpus)

        println("STARTING SCHEDULER WITH: " + numCores + " CPU => " + numGpus + " GPUS")
        for (_ <- 0 until numCores) {
            executorService.submit(new JmlcExecutor(this, "CPU", null))
        }
        for (i <- 0 until numGpus) {
            executorService.submit(new JmlcExecutor(this, "GPU", gCtxs.get(i)))
        }
    }

    def shutdown(): Unit = {
        executorService.shutdown()
    }

    def schedule(executor: JmlcExecutor): Array[SchedulingRequest]

    /**
      * Registers a model with this scheduler. This should be called before enqueueing requests
      * @param model Model object to be registered
      */
    def addModel(model: Model): Unit = {
        modelQueues.putIfAbsent(model.name, new LinkedBlockingDeque[SchedulingRequest]())
        latencyObjectives.putIfAbsent(model.name, model.latencyObjective)
    }

    /**
      * Sets a flag indicating if detailed statistics should be gathered which profile the time spent
      * in various stages of the execution pipeline
      * @param flag Boolean flag indicating whether statistics should be gathered
      */
    def setStatistics(flag: Boolean) = { _statistics = flag }

    def timeout: Duration

    /**
      * Method which is used to update scheduler state of execution of a batch. If necessary
      * objects implementing the Scheduler trait should override this method and implement any logic needed
      * to post-process execution after a batch
      *
      * @param model String indicating the name of the model which was just executed
      * @param latency A measure of latency for this batch
      * @param batchSize The number of examples in the batch
      * @param execType The device type on which the batch was executed
      */
    def onCompleteCallback(model: String, latency: Double, batchSize: Int, execType: String) : Unit

    protected val requestQueue = new LinkedBlockingDeque[SchedulingRequest]()
    protected var modelQueues = new ConcurrentHashMap[String, BlockingQueue[SchedulingRequest]]()
    protected val dummyResponse = PredictionResponse(null, -1, null)
    protected val latencyObjectives = new ConcurrentHashMap[String,Duration]()
    var counter = 0

    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which predictio
      * @return
      */
    private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val statistics = if (_statistics) RequestStatistics() else null
        val schedulingRequest = SchedulingRequest(
            request, model, new CountDownLatch(1), System.nanoTime(), null, statistics)
        statistics.queueSize = modelQueues.get(model.name).size
        requestQueue.add(schedulingRequest)
        modelQueues.get(model.name).add(schedulingRequest)
        counter += 1
        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e : scala.concurrent.TimeoutException => dummyResponse
        }
    }
}

class BasicBatchingScheduler(override val timeout: Duration) extends Scheduler {

    var isGpuEnabled = false
    val modelBatchSizes = new ConcurrentHashMap[String, ConcurrentHashMap[String,Int]]()
    modelBatchSizes.put("GPU", new ConcurrentHashMap[String,Int]())
    modelBatchSizes.put("CPU", new ConcurrentHashMap[String,Int]())

    def getOptimalBatchSize(model : String, execType: String) : Int = {
        modelBatchSizes.get(execType).putIfAbsent(model, 1)
        modelBatchSizes.get(execType).get(model)
    }

    // TODO: Implement these methods to handle heterogeneous models
    def getExpectedExecutionTimeCPU(model : String, batchSize : Int) : Long = { 2L }

    def getExpectedExecutionTimeGPU(model : String) : Long = { 2L }

    def getExpectedExecutionTime(model : String, batchSize : Int) : Long = { if (isGpuEnabled) 2L else 2L }

    def setGpuEnabled(flag: Boolean) : Unit = { isGpuEnabled = flag }

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

    }

    /**
      * Returns a list of requests to execute. If the list contains more than one element, they will be batched
      * by the executor. Returns an empty list when there are no models to be scheduled.
      * @param executor an Executor instance
      * @return a list of model requests to process
      */
    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        if (requestQueue.size() > 0) {
            val execType = executor.getExecType()
            dummyResponse.synchronized {
                if (requestQueue.size() > 0) {
                    val schedulableModels = getSchedulableModels(execType)
                    if (schedulableModels.nonEmpty) {
                        val (nextModel, nextBatchSize) = getNextModelAndBatchSize(
                            schedulableModels, execType)
                        for (_ <- 0 until nextBatchSize) {
                            val next = modelQueues.get(nextModel).poll()
                            assert(next != null, "Something is wrong")
                            ret :+= next
                        }
                    }
                }
            }
        }
        ret
    }

    /**
      * Helper method which gets the next model to schedule and the optimal batchsize
      * @param models A list of models to schedule
      * @return The model to schedule next
      */
    def getNextModelAndBatchSize(models : Iterable[String], execType: String) : (String, Int) = {
        val nextModel = models.map(
            m => ((getExpectedExecutionTime(m, getOptimalBatchSize(m, execType))), m)
        ).minBy(x => x._1)._2

        val nextBatchSize = min(modelQueues.get(nextModel).size(), 
                                getOptimalBatchSize(nextModel, execType))
        (nextModel, nextBatchSize)
    }

    /**
      * Gets a list of models that are eligible to be run. A model is eligible to be run if it
      * has a greater number of requests enqueued than its optimal batch size. Any model which is about
      * to time out (timing out being defined as 10% remainin on its clock) is executed immediately.
      * @return A list of models which may be scheduled
      */
    def getSchedulableModels(execType: String) : Array[String] = {
        var batchableModels = Array[String]()
        var shortFuse = Array[(String,Long)]()
        val keyIterator = modelQueues.keys()
        while (keyIterator.hasMoreElements) {
            val name = keyIterator.nextElement()
            if (modelQueues.get(name).size() > 0) {
                val nextRequest = modelQueues.get(name).peek()
                if (checkShortFuse(nextRequest)) {
                    shortFuse :+= (name, nextRequest.receivedTime - System.nanoTime())
                }
                if (modelQueues.get(name).size() >= getOptimalBatchSize(name, execType)) {
                    batchableModels :+= name
                }
            }
        }

        var shortFuseArray = Array[String]()
        if (shortFuse.nonEmpty) {
            shortFuseArray :+= shortFuse.minBy(x => x._2)._1
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

class NonBatchingScheduler(override val timeout: Duration) extends Scheduler {

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

    override def onCompleteCallback(model: String, latency: Double, batchSize: Int, execType: String): Unit = {}
}
