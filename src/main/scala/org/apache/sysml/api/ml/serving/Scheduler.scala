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

import scala.concurrent.{Await, Future, future}
import scala.concurrent.duration._
import java.util.{HashMap, ArrayList}
import java.util.concurrent.{ExecutorService, BlockingQueue, LinkedBlockingDeque, CountDownLatch, Executors}
import scala.concurrent.ExecutionContext.Implicits.global

case class SchedulingRequest(request:PredictionRequest, model:Model, latch:CountDownLatch, var response:PredictionResponse)

trait JmlcExecutor extends Runnable {
  @volatile protected var _shouldShutdown:Boolean = false
  def scheduler:Scheduler
  def shutdown():Unit = {
    _shouldShutdown = true
  }
  def run():Unit = {
    while(!_shouldShutdown) {
      val requests = scheduler.schedule(this)
      val responses = execute(requests)
      for(i <- 0 until requests.size()) {
        val request = requests.get(i)
        request.response = responses.get(i)
        request.latch.countDown()
      }
    }
  }
  def execute(request:ArrayList[SchedulingRequest]):ArrayList[PredictionResponse]
}

// one task per GPU
class GpuJmlcExecutor(gpuNumber:Int, override val scheduler:Scheduler) extends JmlcExecutor {
  def execute(requests:ArrayList[SchedulingRequest]):ArrayList[PredictionResponse] = {
    // TODO:
    null
  }
}

// one task per core
class CpuJmlcExecutor(override val scheduler:Scheduler) extends JmlcExecutor {
  def execute(requests:ArrayList[SchedulingRequest]):ArrayList[PredictionResponse] = {
    // TODO:
    null
  }
}

trait Scheduler {
  var executorService:ExecutorService = null
  def start(numCores:Int, cpuMemoryBudgetInBytes:Long, gpus:Array[Int]): Unit = {
    val numGpus = if(gpus == null) 0 else gpus.length
    executorService = Executors.newFixedThreadPool(numCores + numGpus)
    for(i <- 0 until numCores) {
      executorService.submit(new CpuJmlcExecutor(this))
    }
    for(i <- 0 until numGpus) {
      executorService.submit(new GpuJmlcExecutor(gpus(i), this))
    }
  }
  def shutdown():Unit = {
    executorService.shutdown();
  }
  def schedule(executor:JmlcExecutor):ArrayList[SchedulingRequest]
  def addModel(model:Model):Unit = {
    modelQueues.put(model.model, new LinkedBlockingDeque[SchedulingRequest])
  }
  def timeout:Duration
  protected val requestQueue:BlockingQueue[SchedulingRequest] = new LinkedBlockingDeque[SchedulingRequest]
  protected val modelQueues:HashMap[String, BlockingQueue[SchedulingRequest]] = new HashMap[String, BlockingQueue[SchedulingRequest]]
  protected val dummyResponse = new PredictionResponse("test-response", "test-format")
  private[serving] def enqueue(request:PredictionRequest, model:Model):Future[PredictionResponse] = future {
    val schedulingRequest = new SchedulingRequest(request, model, new CountDownLatch(1), null)
    dummyResponse.synchronized {
      requestQueue.add(schedulingRequest)
      modelQueues.get(model.model).add(schedulingRequest)
    }
    try {
      schedulingRequest.latch.await(timeout.length, timeout.unit)
      schedulingRequest.response
    } catch {
      case e:scala.concurrent.TimeoutException => dummyResponse
    }
  }
}

class NoBatching(override val timeout:Duration) extends Scheduler {
  override def schedule(executor:JmlcExecutor):ArrayList[SchedulingRequest] = {
    val ret = new ArrayList[SchedulingRequest]
    dummyResponse.synchronized {
      if(requestQueue.size() > 0) {
        val request = requestQueue.take()
        modelQueues.get(request.model.model).remove(request)
        ret.add(request)
      }
    }
    ret
  }
}
