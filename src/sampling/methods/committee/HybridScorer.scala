package sampling.methods.committee

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class HybridScorer(var instanceScorer: Array[InstanceScorer], dictionarySize:Int, embeddingSize:Int) extends InstanceScorer(dictionarySize, embeddingSize) {
  override def score(instance: TextInstance): Double  = {
    val mainScore = instanceScorer.map(scorer=> scorer.score(instance)).sum / instanceScorer.length
    mainScore
  }


  override def add(instance: TextInstance): HybridScorer.this.type = {
    instanceScorer.foreach(subscorer=> subscorer.add(instance))
    this
  }

  override def init(instances: Array[TextInstance]): HybridScorer.this.type = {
    instanceScorer.foreach(scorer=> scorer.init(instances))
    this
  }


}
