package org.apache.sysml.api.ml.serving
import org.apache.sysml.runtime.matrix.data.MatrixBlock

object BatchingUtils {
        def batchRequests(requests: Array[SchedulingRequest]) : MatrixBlock = {
            var res = requests(0).request.data
            for (req <- requests.tail) {
                res = res.append(req.request.data, new MatrixBlock(), false)
            }
            res
        }

        def unbatchRequests(requests: Array[SchedulingRequest],
                            batchedResults: MatrixBlock) : Array[PredictionResponse] = {
            var responses = Array[PredictionResponse]()
            val start = 0
            for (req <- requests) {
                responses :+= PredictionResponse(
                    batchedResults.slice(start, (start + req.request.requestSize)-1), batchedResults.getNumRows)
            }
            responses
        }
}
