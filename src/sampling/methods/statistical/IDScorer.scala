package sampling.methods.statistical

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

abstract class IDScorer(dictionarySize:Int, embeddingSize:Int) extends InstanceScorer(dictionarySize, embeddingSize) {

  var density = Array.fill[Double](dictionarySize)(0d)

  def transpose(matrix: Array[Array[Float]]): Array[Array[Float]] = {

    val main = Array.fill[Array[Float]](dictionarySize)(Array.fill[Float](dictionarySize)(0f))

    for (i <- 0 until dictionarySize) {
      for (j <- 0 until dictionarySize) {
        main(j)(i) = matrix(i)(j)
      }
    }

    main
  }

  def inverse(matrix: Array[Array[Float]]): Array[Array[Float]] = {
    val tr = transpose(matrix)
    val main = Array.fill[Array[Float]](dictionarySize)(Array.fill[Float](dictionarySize)(0f))

    for (i <- 0 until dictionarySize) {
      for (k <- 0 until dictionarySize) {
        var sum = 0f
        for (j <- 0 until dictionarySize) {
          sum += tr(i)(j) * matrix(j)(k)
        }
        main(i)(k) = sum
      }
    }

    main
  }


  def add(instance:TextInstance):this.type ={
    val vector = bowVector(instance.features)
    density = density.zip(vector).map{case(main, item)=> main + item / count}
    count +=1
    this
  }

}
