package helper

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

class LMNode(var item: Int) {

  val startID = "START".hashCode
  val endID = "END".hashCode

  case class LMPosition(node: LMNode, indices: Array[Int], score: Array[Double]){

    def decode(item: String, split: String): String = {
      var str = item.substring(0, 1)
      for (i <- 1 until item.length) {
        if (indices.contains(i)) str = str + split + item(i)
        else str = str + item(i)
      }
      str
    }

    def isStart() = indices.last == 0

    def accomplished(itemLength: Int): Boolean = {
      indices.last == itemLength || nodeHasEnd()
    }

    def nodeHasEnd():Boolean={
      node.hasEnd()
    }

  }

  var nextScore = Map[Int, LMScore]()
  var nextMap = Map[Int, LMNode]()

  def isEmpty():Boolean={
    nextMap.size == 1 || nextMap.isEmpty
  }

  def merge(other:LMNode):this.type ={
    other.nextScore.foreach(pair=>{
      if(nextScore.contains(pair._1)) nextScore(pair._1).incrementBy(pair._2.count)
      else nextScore += pair._1 -> pair._2
    })

    other.nextMap.foreach(pair=>{
      if(nextMap.contains(pair._1)) nextMap(pair._1).merge(pair._2)
      else nextMap += pair._1-> pair._2
    })

    this
  }

  def save(stream:ObjectOutputStream):this.type ={
    stream.writeInt(nextScore.size)
    nextScore.foreach(pair=>{
      stream.writeInt(pair._1)
      stream.writeObject(pair._2)
    })

    nextMap.foreach(pair=>{
      stream.writeInt(pair._1)
      pair._2.save(stream)
    })
    this
  }

  def load(stream:ObjectInputStream):this.type ={
    val size = stream.readInt()
    for(i<-0 until size){
      val id = stream.readInt()
      val score = stream.readObject().asInstanceOf[LMScore]
      nextScore = nextScore + (id-> score)
    }

    for(i<-0 until size){
      val id = stream.readInt()
      val node = new LMNode(id).load(stream)
      nextMap = nextMap + (id-> node)
    }

    this
  }

  def add(next: String): LMNode = {
    val id = next.hashCode
    if (nextScore.contains(id)) {
      nextScore(id).incrementBy(1)
      nextMap(id)
    }
    else {
      val score = new LMScore(1, 0)
      val node = new LMNode(id)
      nextMap = nextMap.updated(id, node)
      nextScore = nextScore.updated(id, score)
      node
    }
  }

  def add(sequence: Array[String]): this.type = {
    var crrNode = this
    sequence.foreach(item => {
      crrNode = crrNode.add(item)
    })

    crrNode.add("END")
    this
  }

  def addSkip(items: Array[String]): this.type = {
    items.foreach(item => {
      add(item)
    })
    this
  }

  def hasEnd():Boolean = nextScore.contains(endID)
  def isStart():Boolean = item != startID
  def endScore() = nextScore(endID)

  def hasNext(id: Int): Boolean = nextScore.contains(id)

  def score(id: Int): Double = {
    if (nextScore.contains(id)) nextScore(id).score
    else 0d
  }

  def node(id: Int): LMNode = {
    nextMap(id)
  }

  def traverse(item: String, window: Int, position: LMPosition): Array[LMPosition] = {
    var positions = Array[LMPosition]()
    val crrIndices = position.indices
    val crrScore = position.score
    val crrIndex = crrIndices.last
    val crrNode = position.node


    if(!position.isStart() && crrNode.hasEnd()){
      val newScore = endScore().score
      positions = positions :+ LMPosition(crrNode, crrIndices, crrScore.tail :+ newScore)
    }

    val maxIndex = math.min(item.length, crrIndex + window)
    for (i <- crrIndex + 1 to maxIndex) {
      val str = item.substring(crrIndex, i)
      val strId = str.hashCode

      if (crrNode.hasNext(strId)) {
        val nextNode = crrNode.node(strId)
        positions = positions :+ LMPosition(nextNode, crrIndices :+ i, crrScore :+ score(strId))
      }
    }

    positions
  }

  def traverse(item: String): Array[LMPosition] = {
    val window = 7
    var nextPositions = traverse(item, window, LMPosition(this, Array(0), Array(1d)))
    var accomplished = Array[LMPosition]()
    val itemLength = item.length

    while (nextPositions.nonEmpty) {
      val newPositions = nextPositions.flatMap(nextPosition => traverse(item, window, nextPosition))
      nextPositions = newPositions.filter(position => !position.accomplished(itemLength))
      accomplished = accomplished ++ newPositions.filter(position => position.accomplished(itemLength))
    }

    accomplished
  }

  def decode(item: String, split: String): Array[String] = {
    val positions = traverse(item)
    positions.map(position => position.decode(item, split))
  }

  def decodeScores(item: String, split: String): Array[(String, Array[Double])] = {
    val positions = traverse(item)
    positions.map(position => (position.decode(item, split), position.score))
  }

  def normalize(): Unit = {
    val total = nextScore.toArray.map(pair=>{
      val lmScore = pair._2
      lmScore.count
    }).sum

    nextScore.toArray.par.foreach(pair=>{
      val lmId = pair._1
      val lmScore = pair._2
      lmScore.normalize(total)
      nextMap(lmId).normalize()
    })


  }
}


object LMNode {

  var rootNode = new LMNode(0)

  rootNode.add(Array("araba", "danmış"))
  rootNode.add(Array("araba", "lardandı"))
  rootNode.add(Array("araba", "ların"))
  rootNode.add(Array("araba", "lardan"))
  rootNode.add(Array("araba", "ları"))
  rootNode.add(Array("araba", "lar"))
  rootNode.add(Array("araba", "dan"))
  rootNode.add(Array("araba", "da"))

  def main(args: Array[String]): Unit = {
    rootNode.decode("arabalardanı", "#").foreach(decodeStr => println(decodeStr))
  }
}