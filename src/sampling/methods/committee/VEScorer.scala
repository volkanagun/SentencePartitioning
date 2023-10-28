package sampling.methods.committee

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

import scala.util.Random


class VEScorer(dictionarySize: Int, embeddingSize: Int, window: Int, committee: Int) extends InstanceScorer(dictionarySize, embeddingSize) {
  case class Count(var total: Long, var score: Double) {
    def increment(): this.type = {
      total += 1
      this
    }

    def normalize(sum: Long): this.type = {
      score = total.toDouble / sum
      this
    }

    def fresh(sum:Long):Double={
      total.toDouble/sum
    }
  }

  case class Node(word: Int) {

    var next = Map[Int, Count]()
    var previous = Map[Int, Count]()


    def score(id:Int):Double={
      if(next.contains(id)) next(id).total / next.map(_._2.total).sum
      else 1d
    }

    def incrementPrevious(previousID: Int): this.type = {
      if (previous.contains(previousID)) previous(previousID).increment()
      else {
        previous = previous.updated(previousID, Count(1, 0))
      }

      this
    }

    def incrementNext(nextID: Int): this.type = {
      if (next.contains(nextID)) next(nextID).increment()
      else {
        next = next.updated(nextID, Count(1, 0))
      }
      this
    }
  }

  var models = Array.fill[Map[Int, Node]](committee)(
    Map[Int, Node](0-> Node(-1)))

  val rand = new Random(42)


  def add(array: Array[Node], previous: Node, node: Node): (Array[Node], Node) = {
    if(previous.word == -1){
      addNew(array, previous, node)
    }
    else{
      (array, previous)
    }
  }

  def addNew(array: Array[Node], previous: Node, node: Node): (Array[Node], Node) = {
    array.find(a => a.word == previous.word) match {
      case Some(foundNode) => {
        foundNode.incrementNext(node.word);
        node.incrementPrevious(foundNode.word)
        (array, foundNode)
      }
      case None => {
        node.incrementPrevious(previous.word)
        previous.incrementNext(node.word)
        (array :+ previous, previous)
      }
    }
  }

  def update(model: Seq[Array[Node]], indice: Int, array: Array[Node]): Seq[Array[Node]] = {
    var updated = model
    if (updated.length <= indice + 1) updated :+ array
    else {
      updated = updated.updated(indice, array)
      updated
    }


  }

  def add(model: Map[Int, Node], instance: TextInstance): Map[Int, Node] = {
    var updated = model

    instance.featureSequence
      .map(items => items.map(st => Node(st)))
      .foreach(seq => {
        seq.sliding(window, 1).foreach { case nodes => {
          var previous = Node(-1)
          nodes.zipWithIndex.foreach { case(node, indice) => {
            updated(indice).incrementNext(node.word)
            updated = updated.updated(indice, previous)
            updated = updated.updated(indice + 1, node)
            previous = node
          }
          }
        }
        }
      })

    updated
  }

  def score(model: Map[Int, Node], index: Int, previous: Int, crr: Int): Double = {
    if(model.contains(index) && model(index).word == previous){
      model(index).score(crr)
    }
    else{
      1d
    }
  }

  def add(textInstance: TextInstance): this.type = {

    val crrIndex = models.map(model => {
      var total = 0d;
      textInstance.featureSequence.foreach(seq => {
        var previous = -1
        seq.sliding(window, 1).foreach(words => {
          var sum = 0d
          words.zipWithIndex.foreach { case (crr, indice) => {
            val crrScore = score(model, indice, previous, crr) / models.length
            sum = sum + crrScore * math.log(crrScore)
            previous = crr
          }}

          total = total + -1.0 / textInstance.featureSequence.length * sum
        })
      })

      (model, total)
    }).zipWithIndex.sortBy(tuple=> tuple._1._2).last._2

    models(crrIndex) = add(models(crrIndex), textInstance)
    this
  }

  //check
  override def score(textInstance: TextInstance): Double = {

    var total = 0d;

    textInstance.featureSequence.foreach(seq => {
      var previous = -1
      seq.sliding(window, 1).foreach(words => {
        var sum = 0d
        models.foreach(model => {
          words.zipWithIndex.foreach { case (crr, indice) => {
            val crrScore = score(model, indice, previous, crr) / models.length
            sum = sum + crrScore * math.log(crrScore)
            previous = crr
          }
          }
        })
        total = total + -1.0 / textInstance.featureSequence.length * sum
      })
    })

    total
  }


  override def init(instances: Array[TextInstance]): VEScorer.this.type = {
    instances.foreach(instance => add(instance))
    this
  }

}
