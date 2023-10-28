package sampling.methods.clustering

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer


class LanguageModelScorer(dictionarySize:Int, embeddingSize:Int, windowLength: Int = 2) extends InstanceScorer(dictionarySize, embeddingSize) {

  var countMap = Map[Int, Double]()
  var sum = 0d;
  def updateWords(words: Seq[Int]): this.type = {
    val key = words.mkString(" ").hashCode
    countMap = countMap.updated(key, countMap.getOrElse(key, 0d) + 1d)
    sum += 1d
    this
  }


  override def add(instance: TextInstance): LanguageModelScorer.this.type = {
    add(instance.featureSequence)
  }

  def add(sequences: Array[Seq[Int]]): this.type = {
    sequences.foreach(crr=>{
        crr.sliding(windowLength, 1)
          .foreach(sequence => {
            updateWords(sequence)
          })
    })

    this
  }


  def perplexity(words:Seq[Int]):Double={
      val slide =  words.sliding(windowLength, 1)
        .toArray

      val score = slide.map(sequence => {
          val previous = sequence.slice(0, windowLength - 1)
          val previousKey = previous.mkString(" ").hashCode
          val previousCount = countMap.getOrElse(previousKey, 1d);
          val mainKey = sequence.mkString(" ").hashCode
          val mainCount = countMap.getOrElse(mainKey, 1d)
          -math.log(mainCount / previousCount)
        }).sum / slide.length

     math.exp(score)
  }

  override def score(instance: TextInstance): Double  = {
    val perplexities = instance
      .featureSequence
      .map(seq=> perplexity(seq))

    perplexities.sum / perplexities.length
  }


  override def init(instances: Array[TextInstance]): LanguageModelScorer.this.type = {
    instances.foreach(instance=>{
      add(instance.featureSequence)
    })
    this
  }


}
