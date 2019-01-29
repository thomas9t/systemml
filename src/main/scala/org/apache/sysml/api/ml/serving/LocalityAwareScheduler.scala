package org.apache.sysml.api.ml.serving

import java.util.concurrent.{ConcurrentHashMap, CountDownLatch}

import scala.concurrent.Future
import scala.math.min
import scala.concurrent.duration.Duration

object ExecutorQueueManager extends Runnable {
    var _shutDown = false
    var _scheduler = LocalityAwareScheduler
    def shutdown(): Unit = { _shutDown = true }

    override def run() : Unit = {
        if (PredictionService.__DEBUG__) println("Hi From Executor Queue Manager!")
        while (!_shutDown) {
            _scheduler.dummyResponse.synchronized {
                val schedulableModels = _scheduler.executorTypes.map(
                    x => _scheduler.getSchedulableModels(x)).reduce(_ union _)
                if (schedulableModels.nonEmpty) {
                    for (m <- schedulableModels) {
                        // every request batch can go to up to three queues

                        // 1. Every batch goes to the global disk queue since the model might get evicted
                        val diskQueues = _scheduler.executorTypes.map(x => _scheduler.globalDiskQueues.get(x))

                        // 2. If the model is cached in memory, then also put it on the cache queue
                        var cacheQueues = Array[BatchQueue]()
                        if (_scheduler.modelManager.isCached(m))
                            cacheQueues = _scheduler.executorTypes.map(x => _scheduler.globalCacheQueues.get(x))

                        // 3. If the model is local to an executor, then put it on the lowest utilizaiton queue
                        val localExecutionQueues = getLocalExecutionQueues(m)
                        val localQueue = if (localExecutionQueues.nonEmpty)
                            Array[BatchQueue](localExecutionQueues.minBy(x => x.getExpectedExecutionTime))
                        else Array[BatchQueue]()

                        val queues = diskQueues ++ cacheQueues ++ localQueue
                        val nextRequest = _scheduler.modelQueues.get(m).peek()
                        queues.foreach ( queue => {
                            val qsize = _scheduler.modelQueues.get(m).size()
                            if (nextRequest ne queue.getPrevRequest(m)) {
                                val nextBatchSize = min(qsize, _scheduler.getOptimalBatchSize(m, queue.getExecType))
                                assert(nextBatchSize > 0, "Something is wrong - batch size should not be zero")
                                if (PredictionService.__DEBUG__)
                                    println("ENQUEUING: " + nextBatchSize + " FOR: " + m + " ONTO: " + queue.getName)
                                val nextBatch = Batch(
                                    nextBatchSize, _scheduler.getExpectedExecutionTime(
                                        m, nextBatchSize, queue.getExecType),
                                    nextRequest.receivedTime - System.nanoTime(), nextRequest.model.name)
                                queue.enqueue(nextBatch)
                                if (PredictionService.__DEBUG__)
                                    println("DONE WITH ENQUEUING")
                            }
                            queue.updatePrevRequest(m, nextRequest) } )
                        }
                    }
                }
            }
        }

    def getLocalExecutionQueues(model: String) : Array[BatchQueue] = {
        val execs = _scheduler.modelManager.getModelLocality(model)
        var queues = Array[BatchQueue]()
        if (execs == null)
            return queues

        _scheduler.modelManager.synchronized({
            for (ix <- 0 until execs.size()) { _scheduler.executorQueues.get(execs.get(ix)) }
        })

        queues
    }
}

object ExecMode extends Enumeration {
    type MODE = Value
    val LOCAL, GLOBAL_MEM, GLOBAL_DISK = Value
}

object LocalityAwareScheduler extends BatchingScheduler {
    var queueManager : Thread = _

    val globalCacheQueues = new ConcurrentHashMap[String, BatchQueue]()
    val globalDiskQueues = new ConcurrentHashMap[String, BatchQueue]()

    override def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        super.start(numCores, cpuMemoryBudgetInBytes, gpus)

        executorTypes.foreach ( x => {
            globalCacheQueues.putIfAbsent(x, new BatchQueue(x, x + "-CACHE"))
            globalDiskQueues.putIfAbsent(x, new BatchQueue(x, x + "-DISK"))
        } )

        queueManager = new Thread(ExecutorQueueManager)
        queueManager.start()
    }

    override def addModel(model: Model): Unit = {
        super.addModel(model)
    }

    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        val localQueue = executorQueues.get(executor)
        val globalDiskQueue = globalDiskQueues.get(executor.getExecType)
        val globalMemQueue = globalCacheQueues.get(executor.getExecType)
        if (localQueue.size() > 0 || globalDiskQueue.size() > 0 || globalMemQueue.size() > 0) {
            dummyResponse.synchronized {
                if (localQueue.size() > 0 || globalDiskQueue.size() > 0 || globalMemQueue.size() > 0) {
                    if (PredictionService.__DEBUG__)
                        println("BEGIN SCHEDULING FOR: " + executor.getName)
                    val execMode = Array[(BatchQueue, ExecMode.MODE)](
                        (localQueue, ExecMode.LOCAL),
                        (globalDiskQueue, ExecMode.GLOBAL_DISK),
                        (globalMemQueue, ExecMode.GLOBAL_MEM)
                    ).filter(x => x._1.size() > 0).maxBy(x => x._1.getExpectedExecutionTime)._2

                    val batch = execMode match {
                        case ExecMode.LOCAL => localQueue.peek()
                        case ExecMode.GLOBAL_MEM => globalMemQueue.peek()
                        case ExecMode.GLOBAL_DISK => globalDiskQueue.peek()
                    }
                    assert(batch != null, "Something is wrong. Batch should not be null!")

                    // now we need to ask the resource manager if there's enough memory to execute the batch
                    val model = modelManager.get(batch.modelName)

                    // If there's enough memory we can actually remove the requests from the queue and
                    // submit them for processing
                    val mqueue = modelQueues.get(batch.modelName)
                    val numToDequeue = min(batch.size, mqueue.size())

                    // if this value is zero there are no more requests and the batch is stale
                    if (numToDequeue == 0) {
                        execMode match {
                            case ExecMode.LOCAL => localQueue.poll()
                            case ExecMode.GLOBAL_DISK => globalDiskQueue.poll()
                            case ExecMode.GLOBAL_MEM => globalMemQueue.poll()
                        }
                    } else {
                        val memReceived = modelManager.tryAllocMem(model.name, batch.size)
                        if (memReceived < 0) {
                            return ret
                        }

                        // now we need to actually remove the request from the queue since it's going to be processed
                        execMode match {
                            case ExecMode.LOCAL => localQueue.poll()
                            case ExecMode.GLOBAL_DISK => globalDiskQueue.poll()
                            case ExecMode.GLOBAL_MEM => globalMemQueue.poll()
                        }

                        // now we can actually take the original requests out of the model queues
                        if (PredictionService.__DEBUG__)
                            println("SCHEDULING: " + numToDequeue + " FOR " + batch.modelName + " ON " + executor.getName)
                        for (_ <- 0 until numToDequeue) {
                            val nextRequest = mqueue.poll()
                            assert(nextRequest != null, "Something is wrong - request should not be null!")

                            nextRequest.memUse = memReceived
                            nextRequest.statistics.execMode = execMode match {
                                case ExecMode.LOCAL => 0
                                case ExecMode.GLOBAL_MEM => 1
                                case ExecMode.GLOBAL_DISK => 2
                                case _ => -1
                            }
                            ret :+= nextRequest
                        }
                        if (PredictionService.__DEBUG__)
                            println("DONE SCHEDULING")
                    }
                }
            }
        }
        ret
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
        }

        modelQueues.get(model.name).add(schedulingRequest)

        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case _ : scala.concurrent.TimeoutException => dummyResponse
        }
    }
}
