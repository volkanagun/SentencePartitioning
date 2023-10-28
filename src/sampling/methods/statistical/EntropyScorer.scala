package sampling.methods.statistical

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

abstract class EntropyScorer(dictionarySize: Int, embeddingSize: Int) extends InstanceScorer(dictionarySize, embeddingSize) {

  var model = Map[Int, Link]()

  case class Link(i: Int, j: Int, var count: Long) {

    var score = 0d

    override def hashCode(): Int = i * 7 + j

    override def equals(obj: Any): Boolean = {
      val other = obj.asInstanceOf[Link]
      other.i == i && other.j == j
    }

    def increment(): this.type = {
      count = count + 1
      this
    }

    def normalize(icount: Long): this.type = {
      score = count.toDouble / icount
      this
    }
  }


  def normalize(): this.type = {
    model.map(_._2).groupBy(_.i).foreach { case (_, iter) => {
      val total = iter.map(link => link.count).sum
      iter.foreach(link => link.normalize(total))
    }
    }

    this
  }

  override def add(textInstance: TextInstance): this.type = {
    val dependencies = textInstance.featureSequence.flatMap(seq => seq.sliding(2, 1).map { case (Seq(s1, s2)) => {
      Link(s1, s2, 1L)
    }
    })

    dependencies.foreach(link => {
      model = model.updated(link.hashCode(), model.getOrElse(link.hashCode(), link)
        .increment())
    })

    count += 1
    this
  }

  def scoreBinary(textInstance: TextInstance): Double = {

    val dependencies = textInstance.featureSequence.flatMap { features => {
      features.sliding(2, 1).map { case Seq(i, j) => Link(i, j, 1L) }
        .toArray
    }}

    val sum = dependencies.par.map(link => {
      val llint = model.getOrElse(link.hashCode(), link);
      math.log(llint.count.toDouble/count)
    }).sum

    -1.0 / textInstance.featureSequence.length * sum
  }


}
