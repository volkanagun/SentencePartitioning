package sampling.adapters

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class MajorityVoting(val array:Array[InstanceScorer], val k:Int,  val kSelectSize:Int,val maxSelectSize:Int, val threshold:Double)  extends ScoreAdapter(maxSelectSize) {

  var averagingAdapters =  array.map(scorer=> new MovingAverage(scorer, k, kSelectSize,   maxSelectSize, threshold))
  val querter = 0.5
  override def filter(instances: Array[TextInstance]): Array[TextInstance] = {

    val scoredSet = averagingAdapters.flatMap(adapter=> adapter.filter(instances))
      .groupBy(_.hashCode()).map(pair=> (pair._1, pair._2.length.toDouble/averagingAdapters.length))

    val selectedSet = scoredSet.filter(pair=> pair._2 >= querter).map(_._1).toSet

    val selections = instances.filter(instance=> selectedSet.contains(instance.hashCode()))

    array.foreach(scorer=> selections.foreach(instance=> scorer.add(instance)))
    count += selections.length

    selections
  }

  override def init(instances: Array[TextInstance]): MajorityVoting.this.type = {
    array.foreach(scorer=> scorer.init(instances))
    this
  }
}
