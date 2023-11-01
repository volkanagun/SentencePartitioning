package sampling.adapters

import sampling.data.TextInstance

abstract class ScoreAdapter(maxSelectSize: Int) extends Serializable {

  var total = 0d
  var count = 0d

  var totalTime = 0d

  def measureTime[T](f: => T): T = {
    val start = System.nanoTime()
    val ret = f
    val end = System.nanoTime()
    totalTime += (end - start).toDouble / 1000
    ret
  }


  def filter(array: Array[TextInstance]): Array[TextInstance]

  def init(array: Array[TextInstance]): this.type

  def update(array: Array[(Double, TextInstance)]): Array[TextInstance] = {

    total += array.map(_._1).sum
    count += array.length
    array.map(_._2)
  }

  def isStop() = {
    count >= maxSelectSize
  }

  def status(): Unit = {
    println("Status: " + count)
  }
}
