package sampling.methods.statistical

import sampling.data.{Instance, TextInstance}
import smile.math.MathEx
import smile.math.MathEx.c
import smile.math.matrix.{Matrix, matrix}


class FisherInformation(dictionarySize: Int, embeddingSize: Int) extends IDScorer(dictionarySize, embeddingSize) {

  var samples = Array[TextInstance]()
  var meanVector: Array[Double] = null
  var varianceVector: Array[Double] = null
  var covarianceMatrix: Array[Array[Double]] = null
  var inverseCovarianceMatrix: Array[Array[Double]] = null
  var fidMatrix: Matrix = null
  var computeMod = 50

  def computeFisherInformationMatrix(instances: Array[TextInstance]): Unit = {
    //transpose(mean_derivative w.r.t. mean) * covariance * mean-derivative w.r.t sigma + 1/2 * trace()
    val instanceMatrix = matrix(instances.map(instance => c(embeddingPositiveVector(instance.featureSequence))))
    meanVector = mean(instanceMatrix.toArray)
    varianceVector = variance(instanceMatrix.toArray, meanVector)
    //covarianceMatrix =  // covariance(instanceMatrix, meanVector)
    inverseCovarianceMatrix = matrix(MathEx.cov(instanceMatrix.toArray)).inverse().toArray
    fidMatrix = matrix(Range(0, embeddingSize).map(i => Range(0, embeddingSize).toArray.map(j => {
      instanceMatrix.toArray.map(compute(inverseCovarianceMatrix, meanVector, varianceVector, _, i, j)).sum
    })).toArray)
  }

  def computeNewFisherMatrixScore(instances: Array[Instance]): Double = {
    val instanceVector = instances.map(instance => embeddingPositiveVector(instance.asInstanceOf[TextInstance].featureSequence))
    val newFidMatrix = Range(0, embeddingSize).map(i => Range(0, embeddingSize).toArray.map(j => {
      instanceVector.map(compute(inverseCovarianceMatrix, meanVector, varianceVector, _, i, j)).sum
    })).toArray
    val inverseFidMatrix = matrix(newFidMatrix).inverse()
    inverseFidMatrix.mm(fidMatrix).trace()
  }

  def mean(instances: Array[Array[Double]]): Array[Double] = {
    val mean = Array.fill[Double](embeddingSize)(0d)

    instances.foldRight[Array[Double]](mean) { case (x, main) => x.zip(main).map(pair => pair._1 * 1f / instances.length + pair._2)}
  }

  def variance(instances: Array[Array[Double]], mean: Array[Double]): Array[Double] = {
    var variances = Array.fill[Double](embeddingSize)(0d)
    instances.foldRight[Array[Double]](variances) { case (x, main) => {
        x.zip(mean).zip(main).map(tuple => {
          tuple._2 + 1f / instances.length * math.pow((tuple._1._1 - tuple._1._2), 2)
        })
      }
      }
  }


  def covariance(instances: Array[Array[Double]], mean: Array[Double]): Array[Array[Double]] = {
    val matrix = Array.fill[Array[Double]](embeddingSize)(Array.fill[Double](embeddingSize)(0d))

    val vectors = instances.map(instance => instance.zip(mean).map(pair => pair._1 - pair._2))
    val multiplications = vectors.flatMap(crr => vectors.map(other => multMatrixVector(crr, other)))
    multiplications.foldRight[Array[Array[Double]]](matrix) { case (x, main) => {
      x.zip(main).map(pair => {
        (pair._1.zip(pair._2).map(subpair => {
          1d / (instances.length - 1) * (subpair._1 + subpair._2)
        }))
      })
    }
    }
  }


  def compute(covarianceInverse: Array[Array[Double]], mean: Array[Double], variance: Array[Double], input: Array[Double], i: Int, j: Int): Double = {
    val derivative1 = input.zip(variance).map { case (x, sigma) => {
      ((math.log(x) - mean(i)) / (sigma * sigma))
    }
    }

    val derivative2 = input.zip(mean).map { case (x, m) => {
      val ln = math.log(x) - m
      val lnsquare = ln * ln
      -count / variance(j) * lnsquare / math.pow(variance(j), 3)
    }
    }

    dot(mult(derivative1, covarianceInverse), derivative2)
  }

  override def score(instance: TextInstance): Double = {
    try {
      val score = computeNewFisherMatrixScore(Array(instance))
      score
    }
    catch{
      case e:Throwable => {
        println("Error in Fisher Matrix inverse")
        println("Error: "+e.getMessage)
        0d
      }
    }
  }


  override def init(instances: Array[TextInstance]): FisherInformation.this.type = {
    computeFisherInformationMatrix(instances)
    count = instances.length
    this
  }


  override def add(instance: TextInstance): FisherInformation.this.type = {
    if (samples.length % computeMod == 0 && samples.nonEmpty) {
      computeFisherInformationMatrix(samples)
      samples = samples.drop(samples.length / 2)
    }

    samples = samples :+ instance
    this
  }

}

object FisherInformation {

  def apply():FisherInformation={
    val array: Array[Array[Double]] = Array(Array(1d, 0d), Array(0d, 1d))
    val mat = matrix(array).inverse()
    new FisherInformation(100,100)
  }

}