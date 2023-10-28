package sampling.methods.nn

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

class BoltzmannScorer(embeddingSize:Int, windowSize:Int, hiddenSize: Int, epsilon: Float = 0.01f) extends InstanceScorer(embeddingSize, embeddingSize) {

  val stateSize = embeddingSize * windowSize
  var weights = Array.fill[Array[Double]](stateSize)(Array.fill[Double](hiddenSize)(random.nextDouble()))
  var energyOld = 0d

  //var visibles = Array.fill[Float](dictionarySize)(random.nextInt(2))
  //var hiddens = Array.fill[Float](dictionarySize)(random.nextInt(2))
  def sigmoidHidden(input: Array[Double], j: Int): Double = {
    val sum = input.zipWithIndex.map { case (value, i) => value * weights(i)(j) }.sum
    (1f / (1f + math.exp(-sum))).toFloat
  }

  def sigmoidVisible(input: Array[Double], i: Int): Double = {
    val sum = input.zipWithIndex.map { case (value, j) => value * weights(i)(j) }.sum
    (1f / (1f + math.exp(-sum))).toFloat
  }

  def sigmoidHidden(input: Array[Double]): Array[Double] = {
    Range(0, hiddenSize).toArray.map(j => {
      sigmoidHidden(input, j)
    })
  }

  def sigmoidVisible(input: Array[Double]): Array[Double] = {
    Range(0, stateSize).toArray.map(i => {
      sigmoidVisible(input, i)
    })
  }

  def reconstruct(input: Array[Double]): Array[Double] = {
    val hidden = sigmoidHidden(input)
    sigmoidVisible(hidden)
  }

  def binarize(array: Array[Double]): Array[Double] = {
    array.map(i => (if (i >= 0.5) 1d else 0d))
  }

  def randomInput(): Array[Double] = {
    Array.fill[Double](stateSize)(random.nextInt(2))
  }

  def simulate(visibleInput: Array[Double], iter: Int): Array[Array[Double]] = {
    var visible = visibleInput
    var array = Array[Array[Double]](visible)
    for (i <- 0 until iter) {
      val hiddenInput = sigmoidHidden(visible)
      visible = sigmoidVisible(hiddenInput)
      array :+= hiddenInput
      array :+= visible
    }

    array
  }

  def energy(input: Array[Double]): Double = {
    val hiddenState = sigmoidHidden(input)
    var crrenergy = (for (i <- 0 until stateSize;
                          j <- 0 until hiddenSize) yield weights(i)(j) * input(i) * hiddenState(j)).sum

    crrenergy = -crrenergy / 2;
    crrenergy
  }

  /**
   * Constructive divergence learning
   *
   * @param input
   */
  def train(input: Array[Double]): Unit = {
    val simulations = simulate(input, 2)
    val length = simulations.length
    val initial = (simulations(0), simulations(1))
    val fantasy = (simulations(length - 1), simulations(length - 2))
    Range(0, stateSize).toArray.par.foreach(i => {
      for (j <- 0 until hiddenSize) {
        weights(i)(j) = weights(i)(j) + epsilon * {
          initial._1(i) * initial._2(j) - fantasy._1(i) * fantasy._2(j)
        }
      }})
  }

  override def score(instance : TextInstance): Double = {
    val inputArray = embeddingVector(instance.featureSequence, windowSize)
    val crrEnergy = energy(inputArray)
    crrEnergy
  }


  override def add(instance: TextInstance): BoltzmannScorer.this.type = {
    val vector = embeddingVector(instance.featureSequence, windowSize)
    train(vector)
    this
  }

  override def init(instances: Array[TextInstance]): BoltzmannScorer.this.type = {
      instances.foreach(instance=>{
        val inputArray = embeddingVector(instance.featureSequence, windowSize)
        train(inputArray)
      })
    this
  }

}

object BoltzmannScorer extends BoltzmannScorer( 100, 10,100, 0.05f) {

  def test(): Unit = {
    val inputs = Array(
      Array(1d, 0d, 0d, 1d),
      Array(0d, 1d, 1d, 0d),
      Array(1d, 0d, 0d, 0d),
      Array(0d, 0d, 0d, 1d)
    );

    for (i <- 0 until 10000) {
      inputs.foreach(input => train(input))

    }

    val in = Array(0d, 1d, 0d, 0d)
    val rec = reconstruct(in);


    println("Reconstruct [0f, 1f, 0f, 0f]: " + rec.mkString("[", ",", "]"))
    println("Energy: " + energy(in))
    Thread.sleep(1000)

  }

  def main(args: Array[String]): Unit = {
    test()
  }

}
