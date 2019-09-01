package org.apache.sysml.api.ml.serving

object SchedulerFactory {
  def getScheduler(schedulerType: String) : Scheduler = {
    schedulerType match {
      case "non-batching"   => NonBatchingScheduler
      case "basic-batching" => BasicBatchingScheduler
      case "locality-aware" => LocalityAwareScheduler
    }
  }
}
