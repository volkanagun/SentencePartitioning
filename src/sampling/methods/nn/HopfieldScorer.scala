package sampling.methods.nn

import sampling.data.TextInstance
import sampling.methods.statistical.IDScorer

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.util.Random

class HopfieldScorer(embeddingSize: Int, windowSize:Int) extends IDScorer(embeddingSize, embeddingSize) {

  var iter = 2
  var epsilon = 0.01
  val stateSize = embeddingSize * windowSize
  val pairs = Range(0, stateSize).flatMap(i => Range(i + 1, stateSize).map(j => (i, j))).toArray
  var weights = Array.tabulate[Double](stateSize, stateSize)((i, j) => {
    if (Random.nextBoolean()) 1.0 else -1.0
  })
  //Array.fill[Array[Double]](embeddingSize)(Array.fill[Double](embeddingSize)(Random.nextDouble()))
  var states = Array.tabulate[Double](stateSize)(_ => if (Random.nextBoolean()) 1.0 else -1.0)
  var energyOld = 0d
  val threshold = 0.05


  def init(input: Array[Double]): this.type = {
    states = activation(input)

    for (i <- 0 until states.length) {
      for (j <- 0 until states.length) {
        if (i != j) {
          weights(i)(j) += epsilon * states(i) * states(j)
        }
        else {
          weights(i)(j) = 0d
        }
      }
    }


    this
  }

  def train(input: Array[Double]): this.type = {
    var yi = input

    for (k <- 0 until iter) {
      var yin = Array.fill[Double](stateSize)(0d)
      for (i <- 0 until embeddingSize) {
        yin(i) = input(i) + dot(yi, weights, i)
      }

      yin = activation(yin)
    }
    this
  }

  def dot(y: Array[Double], w: Array[Array[Double]], i: Int): Double = {
    var sum = 0d;
    for (j <- 0 until embeddingSize) {
      sum += y(j) * w(j)(i)
    }
    sum
  }




  def activation(input: Array[Double]): Array[Double] = {
    input.map(value => (if (value >= threshold) 1d else -1d))
  }


  def energy(input: Array[Double]): Double = {
    val states = activation(input)
    val sum = dot(states, input)

    var crrenergy = pairs.par.map(pp => {
      weights(pp._1)(pp._2) * states(pp._1) * states(pp._2)
    }).sum

    crrenergy = -crrenergy / 2 - sum;
    crrenergy
  }

  def energy(): Double = {
    val newstates = states
    var crrenergy = (for (i <- 0 until stateSize;
                          j <- i + 1 until stateSize) yield weights(i)(j) * newstates(i) * newstates(j)).sum

    crrenergy = -crrenergy / 2;
    crrenergy
  }

  override def score(instance: TextInstance): Double = {
    val input = embeddingVector(instance.featureSequence, windowSize)
    val crrScore = energy(input)
    crrScore
  }


  override def init(instances: Array[TextInstance]): HopfieldScorer.this.type = {
    println("Initializing Hopfield")
    instances.foreach {
      instance => {
        init(embeddingVector(instance.featureSequence, windowSize))
      }
    }

    this
  }

  override def add(instance: TextInstance): HopfieldScorer.this.type = {
    val vector = embeddingVector(instance.featureSequence, windowSize)
    init(vector)
    this
  }


}
