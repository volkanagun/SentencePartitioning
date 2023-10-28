package sampling.methods.statistical

import sampling.data.TextInstance

class BinaryEntropyScorer(dictionarySize:Int, embeddingSize:Int) extends EntropyScorer(dictionarySize, embeddingSize) {


  override def score(instance: TextInstance): Double = {
    scoreBinary(instance)
  }

  override def init(instances: Array[TextInstance]): BinaryEntropyScorer.this.type = {
    instances.foreach(instance=> {
      add(instance)
    })
    this
  }



}
