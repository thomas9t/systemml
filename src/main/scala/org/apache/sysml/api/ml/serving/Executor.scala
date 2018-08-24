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
import java.util.concurrent.LinkedBlockingDeque
import java.util.concurrent.atomic.LongAdder

import org.apache.sysml.runtime.instructions.gpu.context.GPUContext


case class Batch(requests: Array[SchedulingRequest], expectedTime: Long)

class BatchQueue(execType: String) extends LinkedBlockingDeque[Batch] {
    private val expectedExecutionTime = new LongAdder()

    def enqueue(batch: Batch) : Unit = {
        synchronized {
            this.add(batch)
            expectedExecutionTime.add(batch.expectedTime)
        }
    }

    def dequeue() : Batch = {
        if (this.isEmpty)
            return Batch(Array[SchedulingRequest](), -1)
        synchronized {
            val nextBatch = this.poll()
            expectedExecutionTime.add(-1*nextBatch.expectedTime)
            return nextBatch
        }
    }

    def getExpectedExecutionTime : Long = { expectedExecutionTime.longValue() }

    def getExecType : String = { execType }
}

class JmlcExecutor(scheduler: Scheduler, execType: String, gCtx: GPUContext) extends Runnable {
    @volatile protected var _shouldShutdown: Boolean = false

    var prevModel = ""

    def shutdown(): Unit = {
        _shouldShutdown = true
    }

    def getExecType: String = { execType }

    def run(): Unit = {
        while (!_shouldShutdown) {
            val batch = scheduler.schedule(this)
            if (batch.requests.nonEmpty) {
                val responses = execute(batch.requests)
                for ((req, resp) <- batch.requests zip responses) {
                    req.response = resp
                    req.latch.countDown()
                }
                println("")
            }
        }
    }

    def execute(requests: Array[SchedulingRequest]): Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.nonEmpty) {
            println("EXEC1")
            val start = System.nanoTime()
            println("EXEC2")
            val batchedMatrixData = BatchingUtils.batchRequests(requests)
            println("EXEC3")
            val batchingTime = System.nanoTime() - start
            val req = requests(0)
            println("EXEC4")
            val script = req.model.script(execType).clone(false)
            println("EXEC5")
            script.setGpuContext(gCtx)
            println("EXEC6")
            script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
            println("EXEC7")
            val execStart = System.nanoTime()
            println("EXEC8")
            val res = script.executeScript().getMatrixBlock(req.model.outputVarName)
            println("EXEC9")
            val execTime = System.nanoTime() - execStart
            println("EXEC10")
            responses = BatchingUtils.unbatchRequests(requests, res)
            println("EXEC11")
            val stop = System.nanoTime()
            println("EXEC12")
            scheduler.onCompleteCallback(req.model.name, stop - req.receivedTime, requests.length, execType)
            println("EXEC13")
            if (req.statistics != null)
                setStatistics(requests, start, batchingTime, execTime)
            if (prevModel.nonEmpty)
                prevModel = req.model.name
            println("EXEC14")
        }
        responses
    }

    def setStatistics(requests: Array[SchedulingRequest],
                      processingStartTime: Long, batchingTime: Long, execTime: Long): Unit = {
        for (req <- requests) {
            req.statistics.batchingTime = batchingTime
            req.statistics.execType = getExecType
            req.statistics.batchSize = requests.length
            req.statistics.queueWaitTime = processingStartTime - req.receivedTime
        }
    }
}