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
import java.lang.Thread;

import org.apache.sysml.runtime.instructions.gpu.context.{GPUContext, GPUContextPool}
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.api.jmlc.PreparedScript

import scala.concurrent.ExecutionContext
import scala.math.{max, min, floor}

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

    def getExecType(): String

    def run(): Unit = {
        while (!_shouldShutdown) {
            val requests = scheduler.schedule(this)
            if (requests.nonEmpty) {
                val responses = execute(requests)
                for ((req, resp) <- requests zip responses) {
                    resp.execType = this.getExecType
                    req.response = resp
                    req.latch.countDown()
                }
            }
        }
    }

    def execute(request: Array[SchedulingRequest]): Array[PredictionResponse]
}

// one task per GPU
class GpuJmlcExecutor(gCtx: GPUContext, override val scheduler: Scheduler) extends JmlcExecutor {

    def getExecType() = { "GPU" }

    def execute(requests: Array[SchedulingRequest]): Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.length > 0) {
            println("EXEC GPU => " + requests.length)
            val start = System.nanoTime()
            val batchedMatrixData = BatchingUtils.batchRequests(requests)
            val batchingTime = System.nanoTime() - start
            val req = requests(0)
            val script = req.model.scriptGpu.clone(false)
            script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
            val computeStart = System.nanoTime()
            val res = script.executeScript(this.gCtx).getMatrixBlock(req.model.outputVarName)
            val computeTime = System.nanoTime() - computeStart
            val unbatchStart = System.nanoTime()
            responses = BatchingUtils.unbatchRequests(requests, res, batchingTime, computeTime)
            val stop = System.nanoTime()
            scheduler.onCompleteCallback(req.model.name, stop - start, requests.length, "GPU")
            println("DONE EXEC GPU")
        }
        responses
    }
}

// one task per core
class CpuJmlcExecutor(override val scheduler: Scheduler) extends JmlcExecutor {

    def getExecType() = { "CPU" }

    def execute(requests : Array[SchedulingRequest]) : Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.length > 0) {
            println("EXEC CPU => " + requests.length)
            val start = System.nanoTime()
            val batchedMatrixData = BatchingUtils.batchRequests(requests)
            val batchingTime = System.nanoTime() - start
            val req = requests(0)
            val script = req.model.script.clone(false)
            val computeStart = System.nanoTime()
            script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
            val computeTime = System.nanoTime() - computeStart
            val res = script.executeScript().getMatrixBlock(req.model.outputVarName)
            responses = BatchingUtils.unbatchRequests(requests, res, batchingTime, computeStart)
            val stop = System.nanoTime()
            scheduler.onCompleteCallback(req.model.name, stop - start, requests.length, "CPU")
            println("DONE EXEC CPU")
        }
        responses
    }
}


trait Scheduler {
    var executorService: ExecutorService = _
    implicit val ec : ExecutionContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(10000))

    def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        val gCtxs = if (gpus != null) GPUContextPool.reserveAllGPUContexts() else null
        val numGpus = if (gCtxs != null) gCtxs.size() else 0
        executorService = Executors.newFixedThreadPool(numCores + numGpus)

        println("STARTING SCHEDULER WITH: " + numCores + " CPU => " + numGpus + " GPUS")
        for (i <- 0 until numCores) {
            executorService.submit(new CpuJmlcExecutor(this))
        }
        for (i <- 0 until numGpus) {
            executorService.submit(new GpuJmlcExecutor(gCtxs.get(i),this))
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

    def onCompleteCallback(model: String, latency: Double, batchSize: Int, execType: String) : Unit

    protected val requestQueue = new LinkedBlockingDeque[SchedulingRequest]()
    protected var modelQueues = new ConcurrentHashMap[String, BlockingQueue[SchedulingRequest]]()
    protected val dummyResponse = PredictionResponse(null, -1)
    var counter = 0

    private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val schedulingRequest = SchedulingRequest(request, model, new CountDownLatch(1), System.nanoTime())
        println("REQUEST RECEIVED FOR: " + model.name)
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
    val modelBatchSizes = new ConcurrentHashMap[String, ConcurrentHashMap[String,Int]]()
    modelBatchSizes.put("GPU", new ConcurrentHashMap[String,Int]())
    modelBatchSizes.put("CPU", new ConcurrentHashMap[String,Int]())
    // modelExecTimes = new ConcurrentHashMap[String,ConcurrentHashMap[String,RLSEstimator]]()
    // modelExecTimes.put("GPU", new ConcurrentHashMap[String,RLSEstimator]())
    // modelExecTimes.put("CPU", new ConcurrentHashMap[String,RLSEstimator]())

    def getOptimalBatchSize(model : String, execType: String) : Int = {
        modelBatchSizes.get(execType).putIfAbsent(model, 1)
        modelBatchSizes.get(execType).get(model)
    }

    def getExpectedExecutionTimeCPU(model : String, batchSize : Int) : Long = { 2L }

    def getExpectedExecutionTimeGPU(model : String) : Long = { 2L }

    def getExpectedExecutionTime(model : String, batchSize : Int) : Long = { if (isGpuEnabled) 2L else 2L }

    def setGpuEnabled(flag: Boolean) : Unit = { isGpuEnabled = flag }

    override def onCompleteCallback(model: String, 
                                    latency: Double, 
                                    batchSize: Int, 
                                    execType: String): Unit = {
        if (latency != latencyObjective.toNanos) {
            modelBatchSizes.synchronized({
                val prevSize = modelBatchSizes.get(execType).get(model)
                modelBatchSizes.get(execType).put(model,
                    if (latency < latencyObjective.toNanos) prevSize+2 else floor(prevSize*0.90).toInt)
                println("UPDATING " + execType + " BATCH SIZE FOR: " + model + " => " + modelBatchSizes.get(execType).get(model))
            })
        }

    }

    // TODO deal with case that there are more resources available
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
                    println("WARNING: SHORT FUSE!")
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

    def checkShortFuse(request: SchedulingRequest) : Boolean = {
        val elapsed = System.nanoTime() - request.receivedTime
        val shortFuse = elapsed >= (1.2*latencyObjective.toNanos).toLong
        shortFuse
    }

}

class NonBatchingScheduler(override val timeout: Duration,
                           override val latencyObjective: Duration) extends Scheduler {

    println("HELLO FROM NON-BATCHING SCHEDULER")
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
