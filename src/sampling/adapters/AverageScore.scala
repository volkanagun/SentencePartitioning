package sampling.adapters

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class AverageScore(val scorer:InstanceScorer, maxSelectSize:Int) extends ScoreAdapter(maxSelectSize) {
  override def filter(array: Array[TextInstance]):Array[TextInstance]={
    val instances = array.map(instance=> (scorer.score(instance), instance))
      .filter(pair=> !pair._1.isNaN && !pair._1.isInfinite)
    var selected =  instances.filter(pair=> pair._1 >= total/count).map(_._2)
    val topCount = maxSelectSize - count.toInt - selected.length
    update(instances.toArray)
    selected.toArray.take(topCount)
  }

  override def init(array: Array[TextInstance]): AverageScore.this.type = {
    scorer.init(array)
    this
  }
}
