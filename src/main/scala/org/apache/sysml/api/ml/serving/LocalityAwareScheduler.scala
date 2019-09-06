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

import java.util.concurrent.{CountDownLatch}

import scala.concurrent.Future
import scala.math.min

object ExecMode extends Enumeration {
    type MODE = Value
    val LOCAL, GLOBAL_MEM, GLOBAL_DISK = Value
}

object LocalityAwareScheduler extends BatchingScheduler {

    override def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        LOG.info(s"Starting Basic Batching Scheduler with: ${numCores} CPUs and ${gpus} GPUs")
        super.start(numCores, cpuMemoryBudgetInBytes, gpus)
    }

    override def addModel(model: Model): Unit = {
        super.addModel(model)
    }

    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        val execType = executor.getExecType
        dummyResponse.synchronized {
            val schedulableModels = getSchedulableModels(execType)
            if (schedulableModels.nonEmpty) {
                LOG.info("Actually making scheduling decision")
                LOG.info("Schedulable Models: " + schedulableModels.mkString(" "))
                LOG.info("Local Model: " + executor.prevModel)
                LOG.info("Other Models: " + (schedulableModels - executor.prevModel).mkString(" "))
                val localQueueUtilization = if (schedulableModels.contains(executor.prevModel))
                    getExpectedExecutionTime(executor.prevModel) else getExpectedExecutionTime(executor.prevModel)
                LOG.info(s"Local Queue Utilization: ${localQueueUtilization}")
                val otherQueueUtilization = (
                            schedulableModels - executor.prevModel
                        ).map(x => getExpectedExecutionTime(x)).reduce(_ + _)
                LOG.info(s"Other Queue Utilization: ${otherQueueUtilization}")
                val mode = if (localQueueUtilization >= otherQueueUtilization) ExecMode.LOCAL else ExecMode.GLOBAL_MEM
                LOG.info(s"Exec mode: ${mode}")
                val nextModel = if (mode == ExecMode.LOCAL)
                    executor.prevModel else getNextModel(schedulableModels - executor.prevModel, execType)
                LOG.info(s"Next model: ${nextModel}")

                val nextBatchSize = min(modelQueues.get(nextModel).size(),
                    getOptimalBatchSize(nextModel, execType))
                LOG.info(s"Next batch size: ${nextBatchSize}")

                LOG.info(s"Scheduling ${nextBatchSize} requests for ${nextModel} onto ${executor.getName}")
                assert(nextBatchSize > 0, "Something is wrong. Batch size should not be zero")
                for (_ <- 0 until nextBatchSize) {
                    val next = modelQueues.get(nextModel).poll()
                    assert(next != null, "Something is wrong. Next model should not be null")
                    ret :+= next
                }
            }
        }
        ret
    }

    def getNextModel(models : Iterable[String], execType: String) : String = {
        models.map(m =>
            (getOptimalBatchSize(m, execType)*getExpectedExecutionTime(m), m)).minBy(x => x._1)._2
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

        if (_statistics) {
            statistics.queueSize = modelQueues.get(model.name).size
            statistics.preprocWaitTime = System.nanoTime() - request.receivedTime
            statistics.receivedTime = request.receivedTime
        }
        modelQueues.get(model.name).add(schedulingRequest)
        LOG.info(s"Received request for ${model.name}. Queue Size is now: ${statistics.queueSize}")
        counter += 1
        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e: scala.concurrent.TimeoutException => dummyResponse
        }
    }
}
