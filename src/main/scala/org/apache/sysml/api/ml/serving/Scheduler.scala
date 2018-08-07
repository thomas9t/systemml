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

import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool

import scala.concurrent.ExecutionContext
import scala.math.{max, min}

case class SchedulingRequest(request: PredictionRequest,
                             model: Model,
                             latch: CountDownLatch,
                             receivedTime: Long,
                             var response: PredictionResponse = null)

trait JmlcExecutor extends Runnable {
    @volatile protected var _shouldShutdown: Boolean = false

    def scheduler: Scheduler

    def shutdown(): Unit = {
        _shouldShutdown = true
    }

    def run(): Unit = {
        while (!_shouldShutdown) {
            val requests = scheduler.schedule(this)
            if (requests.nonEmpty) {
                val responses = execute(requests)
                val latency = System.nanoTime() - requests(0).receivedTime
                for ((req, resp) <- requests zip responses) {
                    req.response = resp
                    req.latch.countDown()
                }
                scheduler.onCompleteCallback(requests(0).model.name, latency, requests.length)
            }
        }
    }

    def execute(request: Array[SchedulingRequest]): Array[PredictionResponse]
}

// one task per GPU
class GpuJmlcExecutor(override val scheduler: Scheduler) extends JmlcExecutor {
    def execute(requests: Array[SchedulingRequest]): Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.length > 0) {
            val batchedMatrixData = BatchingUtils.batchRequests(requests)
            val req = requests(0)
            val script = req.model.scriptGpu
            script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
            val stret = script.executeScript()
            val res = stret.getMatrixBlock(req.model.outputVarName)
            responses = BatchingUtils.unbatchRequests(requests, res)
        }
        responses
    }
}

// one task per core
class CpuJmlcExecutor(override val scheduler: Scheduler) extends JmlcExecutor {
    def execute(requests : Array[SchedulingRequest]) : Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.length > 0) {
            val batchedMatrixData = BatchingUtils.batchRequests(requests)
            val req = requests(0)
            val script = req.model.script
            script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
            val res = script.executeScript().getMatrixBlock(req.model.outputVarName)
            responses = BatchingUtils.unbatchRequests(requests, res)
        }
        responses
    }
}

trait Scheduler {
    var executorService: ExecutorService = _
    implicit val ec : ExecutionContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(10000))

    def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        val numGpus = if (gpus == null) 0 else gpus.split(",").length
        if (gpus != null)
            GPUContextPool.AVAILABLE_GPUS = gpus

        executorService = Executors.newFixedThreadPool(numCores + numGpus)
        for (i <- 0 until numCores) {
            executorService.submit(new CpuJmlcExecutor(this))
        }
        for (i <- 0 until numGpus) {
            executorService.submit(new GpuJmlcExecutor(this))
        }
    }

    def shutdown(): Unit = {
        executorService.shutdown()
    }

    def schedule(executor: JmlcExecutor): Array[SchedulingRequest]

    def addModel(model: Model): Unit = {
        modelQueues.putIfAbsent(model.name, new LinkedBlockingDeque[SchedulingRequest]())
    }

    def timeout: Duration
    def latencyObjective: Duration

    def onCompleteCallback(model: String, latency: Double, batchSize: Int) : Unit

    protected val requestQueue = new LinkedBlockingDeque[SchedulingRequest]()
    protected var modelQueues = new ConcurrentHashMap[String, BlockingQueue[SchedulingRequest]]()
    protected val dummyResponse = PredictionResponse(null)
    var counter = 0

    private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val schedulingRequest = SchedulingRequest(request, model, new CountDownLatch(1), System.nanoTime())
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

class BasicBatchingScheduler(override val timeout: Duration,
                             override val latencyObjective:  Duration) extends Scheduler {

    var isGpuEnabled = false
    var modelBatchSizes = new ConcurrentHashMap[String, Int]()
    val RLS : RLSEstimator = new RLSEstimator()

    def getOptimalBatchSize(model : String) : Int = {
        modelBatchSizes.putIfAbsent(model, 2)
        modelBatchSizes.get(model)
    }

    def getExpectedExecutionTimeCPU(model : String, batchSize : Int) : Long = { 2L }

    def getExpectedExecutionTimeGPU(model : String) : Long = { 2L }

    def getExpectedExecutionTime(model : String, batchSize : Int) : Long = { if (isGpuEnabled) 2L else 2L }

    def setGpuEnabled(flag: Boolean) : Unit = { isGpuEnabled = flag }

    override def onCompleteCallback(model: String, latency: Double, batchSize: Int): Unit = {
        if (latency != latencyObjective.toNanos) {
            modelBatchSizes.synchronized({
                val prevSize = modelBatchSizes.get(model)
                modelBatchSizes.put(model,
                    if (latency < latencyObjective.toNanos) prevSize+2 else max((prevSize*.90).toInt, 1))
            })
        }
        //RLS.enqueueExample(batchSize, latency)
    }

    // TODO deal with case that there are more resources available
    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        if (requestQueue.size() > 0) {
            dummyResponse.synchronized {
                if (requestQueue.size() > 0) {
                    val schedulableModels = getSchedulableModels()
                    if (schedulableModels.nonEmpty) {
                        val (nextModel, nextBatchSize) = getNextModelAndBatchSize(schedulableModels)
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
    def getNextModelAndBatchSize(models : Iterable[String]) : (String, Int) = {
        val nextModel = models.map(
            m => ((getExpectedExecutionTime(m, getOptimalBatchSize(m))), m)
        ).minBy(x => x._1)._2

        val nextBatchSize = min(modelQueues.get(nextModel).size(), getOptimalBatchSize(nextModel))
        (nextModel, nextBatchSize)
    }

    /**
      * Gets a list of models that are eligible to be run. A model is eligible to be run if it
      * has a greater number of requests enqueued than its optimal batch size. Any model which is about
      * to time out (timing out being defined as 10% remainin on its clock) is executed immediately.
      * @return A list of models which may be scheduled
      */
    def getSchedulableModels() : Array[String] = {
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
                if (modelQueues.get(name).size() >= getOptimalBatchSize(name)) {
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

    def checkShortFuse(request: SchedulingRequest) : Boolean = {
        val elapsed = System.nanoTime() - request.receivedTime
        val shortFuse = elapsed >= (0.9*timeout.toNanos).toLong
        shortFuse
    }

}

class NonBatchingScheduler(override val timeout: Duration,
                           override val latencyObjective: Duration) extends Scheduler {
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

    override def onCompleteCallback(model: String, latency: Double, batchSize: Int): Unit = {}
}
