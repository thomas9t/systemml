package org.apache.sysml.api.ml.serving
import org.apache.sysml.runtime.matrix.data.MatrixBlock

object BatchingUtils {
        def batchRequests(requests: Array[SchedulingRequest]) : MatrixBlock = {
            val execStartTime = System.nanoTime()
            if (requests.length == 1) {
                requests(0).queueWaitTime = execStartTime - requests(0).receivedTime
                return requests(0).request.data
            }

            val ncol = requests(0).request.data.getNumColumns
            val res = new MatrixBlock(requests.length, ncol, -1).allocateDenseBlock()
            val doubles = res.getDenseBlockValues
            var start = 0
            for (req <- requests) {
                req.queueWaitTime = execStartTime - req.receivedTime
                System.arraycopy(req.request.data.getDenseBlockValues, 0, doubles, start, ncol)
                start += ncol
            }
            res
        }

        def unbatchRequests(requests: Array[SchedulingRequest],
                            batchedResults: MatrixBlock,
                            execTime: Long,
                            batchingTime: Long) : Array[PredictionResponse] = {
            var responses = Array[PredictionResponse]()
            val start = 0
            for (req <- requests) {
                val unbatchStart = System.nanoTime()
                val resp = PredictionResponse(batchedResults.slice(
                    start, (start + req.request.requestSize)-1), 
                    batchedResults.getNumRows)
                resp.unbatchingTime = System.nanoTime() - unbatchStart
                resp.batchingTime = batchingTime
                resp.execTime = execTime
                resp.queueWaitTime = req.queueWaitTime
                resp.queueSize = req.queueSize
                responses :+= resp
            }

            responses
        }
}
