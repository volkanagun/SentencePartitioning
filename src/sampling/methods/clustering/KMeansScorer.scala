package sampling.methods.clustering

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class KMeansScorer(dictionarySize:Int, embeddingSize:Int, clusterSize: Int, val k: Int) extends InstanceScorer(dictionarySize, embeddingSize) {

  var centers = Array.fill[Array[Double]](clusterSize)(Array.tabulate[Double](dictionarySize)(_ => random.nextDouble()))

  def similarity(array: Array[Double]): Array[Double] = {
    centers.map(center => eucledian(array, center))
  }

  def average(center: Array[Double], array: Array[Double]): Array[Double] = {
    center.zip(array).map { case (a, b) => a + 1d / 1000 * b }
  }

  override def add(instance:TextInstance): this.type = {
    val array = bowVector(instance.features)
    similarity(array).zipWithIndex.sortBy(_._1).take(k).foreach { case (score, index) => {
      centers(index) = average(centers(index), array)
    }}
    count = count + 1
    this
  }


  def append(instance: TextInstance): this.type = {
    val array = bowVector(instance.features)
    similarity(array).zipWithIndex.sortBy(_._1).take(k).foreach { case (score, index) => {
      centers(index) = average(centers(index), array)
    }
    }
    this
  }


  override def init(instances: Array[TextInstance]): KMeansScorer.this.type = {
    instances.foreach(instance=> append(instance))
    this
  }

  override def score(instance: TextInstance): Double  = {
    val array = bowVector(instance.features)
    val similarities = similarity(array).sorted
    val sum = similarities.sum
    val entropy = -similarities.map(k=> k/sum * math.log(k/sum)).sum
    entropy
  }

}
