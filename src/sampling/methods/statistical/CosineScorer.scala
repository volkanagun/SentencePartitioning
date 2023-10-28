package sampling.methods.statistical

import sampling.data.TextInstance

class CosineScorer(dictionarySize:Int, embeddingSize:Int) extends IDScorer(dictionarySize, embeddingSize) {

  override def score(textInstance: TextInstance): Double = {
    val inputVector = bowVector(textInstance.features)
    val scr =  1.0 / cosine(density,inputVector)
    scr
  }

  override def init(instances: Array[TextInstance]): CosineScorer.this.type = {
    instances.foreach(instance=>  add(instance))
    this
  }


}
