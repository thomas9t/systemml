package org.apache.sysml.api.ml.serving
import scala.concurrent.duration._

object SchedulerType extends Enumeration {
    type SchedulerType
    val NON_BATCHING, BASIC_BATCHING, LOCALITY_AWARE = Value
}

import SchedulerType._

object SchedulerFactor {

    def createScheduler(stype: SchedulerType, timeout: Duration) : Scheduler = {
        stype match {
            case NON_BATCHING => new NonBatchingScheduler(timeout)
            case BASIC_BATCHING => new BasicBatchingScheduler(timeout)
            case LOCALITY_AWARE => new LocalityAwareScheduler(timeout)
            case _ => throw new RuntimeException("Invalid scheduler type")
        }
    }

}