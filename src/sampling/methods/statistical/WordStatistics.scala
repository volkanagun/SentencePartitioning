package sampling.methods.statistical

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class WordStatistics(dictionarySize:Int, embeddingSize:Int) extends InstanceScorer(dictionarySize, embeddingSize){
  var vocabulary = Map[Int, Int]()
  var totalWords = 0
  var distictWords = 0

  override def score(instance: TextInstance): Double = {
   val totalFrequency = instance.features.map{case(ww,_) => vocabulary.getOrElse(ww, 0)}.sum
    totalWords/totalFrequency
  }


  override def add(instance: TextInstance): WordStatistics.this.type = {
    instance.features.foreach{case(ii, count)=>{
      vocabulary = vocabulary.updated(ii, vocabulary.getOrElse(ii, 0) + count.toInt)
      totalWords += count.toInt
      distictWords += (if(vocabulary.contains(ii)) 0 else 1)
    }}
    this
  }

  override def init(instances: Array[TextInstance]): WordStatistics.this.type = {
    instances.foreach(instance=> add(instance))
    this
  }
}
