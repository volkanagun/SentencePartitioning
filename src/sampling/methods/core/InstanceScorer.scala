package sampling.methods.core

import data.Dataset
import sampling.data.TextInstance

import java.util.Locale
import scala.collection.IterableOnce.iterableOnceExtensionMethods
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable}
import scala.util.Random

abstract class InstanceScorer(val dictionarySize: Int, val embeddingSize: Int) extends Serializable {
  var count = 1d
  var total = 0d

  val random = new Random(42)
  val locale = new Locale("tr")
  lazy val embeddings = Dataset.loadHuewei().embeddingMap.map(pair => pair._1.toLowerCase().hashCode -> pair._2.array)
  var weighting = Map[Int, Array[Double]]()
  val dummy = Array.fill[Double](embeddingSize)(random.nextDouble())

  def postCompute():this.type  = this

  def score(instance: TextInstance): Double

  def add(instance: TextInstance): this.type

  def init(instances: Array[TextInstance]): this.type


  def randomVector(index: Int): Array[Double] = {

    if (embeddings.contains(index)) {
      embeddings(index)
    }
    else {
      dummy
    }

  }

  def embeddingVector(sequence: Array[Seq[Int]]): Array[Double] = {
    val map = bowMap(sequence)
    val vectors = map.filter{case(index, _)=>{embeddings.contains(index)}}.map{case(index, _)=> embeddings(index)}
      .toArray :+ dummy
    val init = Array.fill[Double](embeddingSize)(0d)
    val finalVector = vectors.foldRight[Array[Double]](init){case(crr, main) => {
      main.zip(crr).map(pair=> pair._1 + pair._2 / vectors.length)
    }}
    finalVector
  }

  def average(sequence:Seq[Array[Double]]):Array[Double]={
    var main = Array.fill[Double](embeddingSize)(0d)
    sequence.foreach(item=>{
      main = main.zip(item).map{case(m, i)=> m + i/sequence.length}
    })
    main
  }

  def embeddingVector(sequence: Array[Seq[Int]], size:Int): Array[Double] = {
    val indexSeq = sequence.head
    val vectorSeq = indexSeq.map(index=> embeddings.getOrElse(index, dummy))
    val subSize = indexSeq.length / size + 1
    var result = vectorSeq.sliding(subSize, subSize).toArray.flatMap(items=>{
        average(items)
    })

    val restSize = math.max(size * embeddingSize - result.length, 0)
    result = result ++ Array.fill[Double](restSize)(0d)

    result
  }

  def embeddingPositiveVector(sequence: Array[Seq[Int]]): Array[Double] = {
    val seqmatrix = (bowMap(sequence))
    val vector = multMatrix(seqmatrix, embeddings)
    vector
  }

  def embeddingVector(sequence: Map[Int, Double]): Array[Double] = {
    val vector = multMatrix(sequence, embeddings)
    val normResult = normVector(vector)
    normResult
  }

  def embeddingPositiveVector(sequence: Map[Int, Double]): Array[Double] = {
    val vector = multMatrix(sequence, embeddings)
    val normResult = normVector(vector)
    normResult
  }

  def bowMap(sequence: Array[Seq[Int]]): Map[Int, Double] = {

    var array = Map[Int, Double]()
    sequence.foreach(subset => {
      subset.foreach(i => {
        array = array.updated(i, 1d)
      })
    })

    array
  }

  def bowVector(sequence: Array[Seq[Int]]): Array[Double] = {
    var array = Array.fill[Double](dictionarySize)(0d)
    sequence.foreach(subset => {
      subset.foreach(i => {
        if(i < dictionarySize)  array(i) = 1d
        else array(dictionarySize - 1) = 1d
      })
    })
    array

  }

  def bowVector(dictionary: Map[Int, Double]): Array[Double] = {
    var array = Array.fill[Double](dictionarySize)(0d)
    var maxindex = dictionarySize - 1;
    dictionary.foreach(pair => {
      val idIndex = pair._1
      if(idIndex < dictionarySize)  array(idIndex) = 1d
      else array(maxindex) = 1d
    })

    array
  }

  def cosine(array: Array[Double], vector: Array[Double]): Double = {
    val nom = array.zip(vector).map(pair => pair._1 * pair._2).sum
    val denom = math.sqrt(array.map(f => f * f).sum) + math.sqrt(vector.map(f => f * f).sum)
    nom / (denom + Float.MinPositiveValue)
  }

  def eucledian(array: Array[Double], vector: Array[Double]): Double = {
    val sim = math.sqrt(array.zip(vector).par.map(pair => Math.pow(pair._1 - pair._2, 2)).sum)
    sim
  }

  def trace(matrix: Array[Array[Double]]): Double = {
    var sum = 0d
    for (i <- 0 until matrix.length) {
      sum += matrix(i)(i)
    }

    sum
  }

  def divideVector(array: Array[Double], scalar: Int): Array[Double] = {
    array.map(_ / scalar)
  }

  def normVector(array: Array[Double]): Array[Double] = {
    val divideBy = math.sqrt(array.map(i => i * i).sum)
    array.map(i => i / divideBy)
  }

  def multMatrix(array: Map[Int, Double], other: Map[Int, Array[Double]]): Array[Double] = {
    val colSize = other.head._2.length
    val result = Array.fill[Double](colSize)(0f)

    for (j <- (0 until colSize).par) {
      array.foreach {
        case (index, value) => {
          result(j) += value * randomVector(index)(j) / array.size
        }
      }
    }

    result
  }

  def multDense(array: Array[Double], other: Array[Array[Double]]): Array[Double] = {
    val colSize = other.head.length
    val result = Array.fill[Double](colSize)(0f)

    for (j <- (0 until colSize)) {
      array.zipWithIndex.foreach {
        case (value, index) => {
          result(j) += value * other(index)(j) / array.size
        }
      }
    }

    result
  }

  def multMatrixVector(array: Array[Double], other: Array[Double]): Array[Array[Double]] = {
    val result = Array.fill[Array[Double]](array.length)(Array.fill[Double](other.length)(0f))

    for (i <- 0 until array.length) {
      for (j <- 0 until other.length) {
        result(i)(j) = array(i) * other(j)
      }
    }

    result
  }


  def mult(array: Array[Double], matrix: Array[Array[Double]]): Array[Double] = {
    val rowSize = array.length
    val colSize = matrix.head.length
    var result = Array.fill[Double](colSize)(0d)

    for (j <- 0 until colSize) {

      for (i <- 0 until rowSize) {
        result(j) += array(i) * matrix(i)(j)
      }

    }

    result
  }

  /*  def mult(array: Array[Double], matrix: Array[Array[Double]], j: Int): Array[Double] = {
      val rowSize = array.length
      val colSize = matrix.head.length
      var result = Array.fill[Double](rowSize)(0d)

      for (i <- 0 until rowSize) {
        result(i) += array(i) * matrix(i)(j)
      }

      result
    }*/

  def multDense(weights: Array[Array[Double]], matrix: Array[Array[Double]]): Array[Array[Double]] = {
    val rowSize = weights.length
    val colSize = matrix.head.length
    val result = Array.fill[Array[Double]](rowSize)(Array.fill[Double](colSize)(0f))

    for (row <- 0 until rowSize) {
      for (column <- 0 until colSize) {
        for (item <- 0 until rowSize) {
          result(row)(column) += weights(row)(item) * matrix(item)(column)
        }
      }
    }

    result
  }

  def addMatrix(matrix1: Array[Array[Double]], matrix2: Array[Array[Double]]): Array[Array[Double]] = {
    matrix1.zipWithIndex.map {
      case (row, i) => row.zipWithIndex.map { case (value, j) => value + matrix2(i)(j) }
    }
  }

  def addVector(vector: Array[Double], other: Array[Double]): Array[Double] = {
    vector.zip(other).map { case (v1, v2) => v1 + v2 }
  }

  def multVector(input: Array[Double], matrix: Array[Array[Double]]): Array[Array[Double]] = {
    val dictionarySize = input.length
    val result = Array.fill[Array[Double]](dictionarySize)(Array.fill[Double](dictionarySize)(0f))

    for (row <- 0 until dictionarySize) {

      for (item <- 0 until dictionarySize) {

        var sum = 0d
        for (column <- 0 until dictionarySize) {
          sum += input(row) * matrix(column)(item)
        }

        result(row)(item) = sum
      }
    }

    result
  }

  def dot(array: Array[Double], matrix: Array[Double]): Double = {
    array.zip(matrix).map { case (a, b) => a * b }.sum
  }

  def mult(array: Array[Double], scalar: Double): Array[Double] = {
    array.map(a => a * scalar)
  }

  def add(array: Array[Double], scalar: Double): Array[Double] = {
    array.map(a => a - scalar)
  }

}
