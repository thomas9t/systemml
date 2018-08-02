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
import scala.collection.concurrent.Map
import java.util.concurrent._

import scala.concurrent.ExecutionContext.Implicits.global
import scala.math.min

case class SchedulingRequest(request: PredictionRequest,
                             model: Model,
                             latch: CountDownLatch,
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
            val responses = execute(requests)
            for ((req, resp) <- requests zip responses) {
                req.response = resp
                req.latch.countDown()
                println(req.latch.toString)
            }

        }
    }

    def execute(request: Array[SchedulingRequest]): Array[PredictionResponse]
}

// one task per GPU
class GpuJmlcExecutor(gpuNumber: Int, override val scheduler: Scheduler) extends JmlcExecutor {
    def execute(requests: Array[SchedulingRequest]): Array[PredictionResponse] = {
        // TODO:
        null
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

    def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: Array[Int]): Unit = {
        val numGpus = if (gpus == null) 0 else gpus.length
        executorService = Executors.newFixedThreadPool(numCores + numGpus)
        for (i <- 0 until numCores) {
            executorService.submit(new CpuJmlcExecutor(this))
        }
        for (i <- 0 until numGpus) {
            executorService.submit(new GpuJmlcExecutor(gpus(i), this))
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

    protected val requestQueue = new LinkedBlockingDeque[SchedulingRequest]()
    protected var modelQueues = new ConcurrentHashMap[String, BlockingQueue[SchedulingRequest]]()
    protected val dummyResponse = PredictionResponse(null)

    private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val schedulingRequest = SchedulingRequest(request, model, new CountDownLatch(1), null)

        requestQueue.add(schedulingRequest)
        modelQueues.get(model.name).add(schedulingRequest)

        println("Added request for model: " + model.name)
        println("Queue size is now: " + modelQueues.get(model.name).size())
        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e : scala.concurrent.TimeoutException => dummyResponse
        }
    }
}

class BasicBatchingScheduler(override val timeout : Duration) extends Scheduler {

    var isGpuEnabled = false

    def getOptimalBatchSize(model : String) : Int = { 2 }

    def getExpectedExecutionTimeCPU(model : String, batchSize : Int) : Long = { 2L }

    def getExpectedExecutionTimeGPU(model : String) : Long = { 2L }

    def getExpectedExecutionTime(model : String, batchSize : Int) : Long = { if (isGpuEnabled) 2L else 2L }

    def setGpuEnabled(flag: Boolean) : Unit = { isGpuEnabled = flag }

    // TODO deal with case that there are more resources available
    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        if (requestQueue.size() > 0) {
            dummyResponse.synchronized {
                if (requestQueue.size() > 0) {
                    val batchableModels = getModelNames().filter(m => modelQueues.get(m).size() >= getOptimalBatchSize(m))
                    if (batchableModels.nonEmpty) {
                        val (nextModel, nextBatchSize) = getNextModelAndBatchSize(batchableModels)
                        for (_ <- 0 until nextBatchSize) {
                            val next = modelQueues.get(nextModel).poll()
                            requestQueue.remove(next)
                            assert(next != null, "Something is wrong")
                            ret :+= next
                        }
                    }
                }
            }
        }
        ret
    }

    def getNextModelAndBatchSize(models : Iterable[String]) : (String, Int) = {
        val nextModel = models.map(
            m => ((getExpectedExecutionTime(m, getOptimalBatchSize(m))), m)
        ).minBy(x => x._1)._2

        val nextBatchSize = min(modelQueues.get(nextModel).size(), getOptimalBatchSize(nextModel))
        (nextModel, nextBatchSize)
    }

    def getModelNames() : Array[String] = {
        var names = Array[String]()
        val keyIterator = modelQueues.keys()
        while (keyIterator.hasMoreElements)
            names :+= keyIterator.nextElement()
        names
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
}
