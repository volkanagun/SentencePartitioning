package sampling.methods.statistical

import sampling.data.TextInstance

class EuclideanScorer(dictionarySize:Int, embeddingSize:Int) extends IDScorer(dictionarySize, embeddingSize) {
  override def score(instance: TextInstance): Double = {
    val score = eucledian(bowVector(instance.features), density)
    score
  }

  override def init(instances: Array[TextInstance]): EuclideanScorer.this.type = {
    instances.foreach(instance=> add(instance))
    this
  }
}
