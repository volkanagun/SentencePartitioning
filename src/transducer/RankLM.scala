package transducer

import experiments.Params
import helper.LMDictionary

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.Locale
import java.util.concurrent.locks.ReentrantLock
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable}
import scala.io.Source

case class RankLink(var key: Int, var target: Int, var count: Long = 1L, var score: Double = 0d) {

  def save(outputStream: ObjectOutputStream): this.type = {
    outputStream.writeInt(key)
    outputStream.writeInt(target)
    outputStream.writeLong(count)
    outputStream.writeDouble(score)
    this
  }

  def load(outputStream: ObjectInputStream): this.type = {
    key = outputStream.readInt()
    target = outputStream.readInt()
    count = outputStream.readLong()
    score = outputStream.readDouble()
    this
  }

  def increment(): this.type = {
    count += 1
    this
  }

  def incrementBy(value:Long): this.type = {
    count += value
    this
  }

  def normalize(total: Long): this.type = {
    score = count.toDouble / total
    this
  }

  override def hashCode(): Int = target.hashCode()

  override def equals(obj: Any): Boolean = {
    val other = obj.asInstanceOf[RankLink]
    other.key == key &&  other.target == target
  }
}
case class RankNode(var partition:String) {
  var split = "[\\$\\#]+"

  def sub():Array[String]={
    partition.split(split)
  }

  def linkPairs(otherPartition:RankNode) : Array[(String, String)] = {
    val crrItems = sub()
    otherPartition.sub().flatMap(other=> crrItems.map(crr=> (crr, other)))
  }

}
case class RankState(var wordId: Int){
  var targetForward = Map[Int, RankLink]()
  var targetBackward = Map[Int, RankLink]()


  def hasForward() = targetForward.nonEmpty
  def hasBackward() = targetBackward.nonEmpty

  def copy():RankState={
    val newState = new RankState(wordId)
    newState.targetForward = targetForward
    newState.targetBackward = targetBackward
    this
  }

  def merge(other:RankState):this.type ={
    other.targetForward.foreach{case(key, link)=>{
      if(targetForward.contains(key)){
        targetForward(key).incrementBy(link.count)
      }
      else{
        targetForward = targetForward + (key->link)
      }
    }}

    other.targetBackward.foreach{case(key, link)=>{
      if(targetBackward.contains(key)){
        targetBackward(key).incrementBy(link.count)
      }
      else{
        targetBackward = targetBackward + (key->link)
      }
    }}

    this

  }

  def addForwardDependency(key: Int, link: RankLink): Unit = {
    if (targetForward.contains(key)) targetForward(key).increment()
    else targetForward = targetForward + (key -> link)
  }

  def addBackwardDependency(key: Int, link: RankLink): Unit = {
    if (targetBackward.contains(key)) targetBackward(key).increment()
    else targetBackward = targetBackward + (key -> link)
  }

  def save(objectStream: ObjectOutputStream): this.type = {
    objectStream.writeInt(wordId)
    objectStream.writeInt(targetForward.size)
    targetForward.foreach(pair => {
      pair._2.save(objectStream)
    })
    objectStream.writeInt(targetBackward.size)
    targetBackward.foreach(pair => {
      pair._2.save(objectStream)
    })
    this
  }

  def load(objectStream: ObjectInputStream): this.type = {
    wordId = objectStream.readInt()
    val size1 = objectStream.readInt()
    for (i <- 0 until size1) {
      val link = RankLink(0, 0).load(objectStream)
      targetForward = targetForward + (link.key -> link)
    }
    val size2 = objectStream.readInt()
    for (i <- 0 until size2) {
      val link = RankLink(0, 0).load(objectStream)
      targetBackward = targetBackward + (link.key -> link)
    }
    this
  }

  def normalize(): this.type = {
    val forwardTotal = targetForward.map(target => target._2.count).sum
    targetForward.foreach(pair => pair._2.normalize(forwardTotal))

    val backwardTotal = targetBackward.map(target => target._2.count).sum
    targetBackward.foreach(pair => pair._2.normalize(backwardTotal))

    this
  }

  def prune(keepSize:Int):this.type ={
    if(!wordId.equals("START".hashCode) && !wordId.equals("END".hashCode)) {

      targetForward = targetForward
        .map(pair => (pair._1, pair._2, pair._2.score)).toArray.sortBy(_._3).reverse
        .take(keepSize).map(tuple => (tuple._1 -> tuple._2)).toMap

      targetBackward = targetBackward
        .map(pair => (pair._1, pair._2, pair._2.score)).toArray.sortBy(_._3).reverse
        .take(keepSize).map(tuple => (tuple._1 -> tuple._2)).toMap

    }
    this

  }

  def forwardLog(key: Int): Double = {
    if (targetForward.contains(key)) {
      -Math.log(targetForward(key).score)
    }
    else {
      0
    }
  }

  def backwardLog(key: Int): Double = {
    if (targetBackward.contains(key)) {
      -Math.log(targetBackward(key).score)
    }
    else {
      0
    }
  }

  override def toString = s"RankState($wordId)"
}

class RankLM(params: Params) extends AbstractLM(params) {

  val skip = "@>"
  val skipBack = "<@"
  val eps = "EPS"
  val modelFilename = s"${parent}/resources/transducers/rank${params.lmID()}.bin"
  var states = Map[Int, RankState]()
  var dictionary = new LMDictionary()


  override def initialize(): RankLM.this.type = {
    if (dictionary.exists()) {
      dictionary.load().normalize()
    }
    else {
      dictionary.fromDictionary(params)
      dictionary.fromText(params)
      dictionary.normalize().save()
    }
    this
  }

  def setDictionary(lmDictionary: LMDictionary):this.type ={
    this.dictionary = lmDictionary
    this
  }

  override def isEmpty(): Boolean = {
    dictionary.isEmpty()
  }
  def setLM(lm: TransducerLM): this.type = {
    this.lm = lm
    this
  }

  def addState(word: String): RankState = {
    val wordId = word.hashCode
    if (states.contains(wordId)) states(wordId)
    else {
      val state = new RankState(wordId)
      states = states + (wordId -> state)
      state
    }
  }

  def count(input: Array[Array[String]]): this.type = {

    input.zipWithIndex.foreach(item => {
      val crrIndex = item._2
      val crrCombination = item._1

      crrCombination.foreach(ngram => {
        val ngramSplits = ngram.split(split)
        ngramSplits.foreach(crrSplit => {
          val crrState = addState(crrSplit)

          for (k <- (crrIndex + 1) until input.length) {
            val skipCombination = input(k)
            val skipDistance = partition(Math.abs(crrIndex - k))
            //val skipBackward = crrSplit + skipBack + skipDistance
            //val skipBackwardId = skipBackward.hashCode
            skipCombination.foreach(skipngram => {
              skipngram.split(split).foreach(skipSplit => {
                val nextState = addState(skipSplit)
                val skipForward = skipSplit + skip + skipDistance
                val skipForwardId = skipForward.hashCode
                crrState.addForwardDependency(skipForwardId, RankLink(skipForwardId, nextState.wordId))
                //nextState.addBackwardDependency(skipBackwardId, RankLink(skipBackwardId, nextState.wordId))
              })
            })
          }
        })
      })
    })

    this
  }

  def partitioning(token:String):Array[(String, Double)]={
    dictionary.filter(dictionary.inference(token), params.lmTopSplit)
  }

  override def splitToken(token: String): Array[String] = partitioning(token).map(_._1)

  def count(sentence: String): this.type = {
    val adjust = "START " + sentence + " END"
    val sequences = adjust.split("\\s+")
      .filter(item => item.matches("\\p{L}{2,}"))
      .sliding(params.lmWindowLength, params.lmWindowLength - 1)
      .toArray

    sequences.foreach(sequence => {
      val combinations = sequence.map(token => partitioning(token)
        .map(_._1))
      count(combinations)
    })

    this
  }

  def rank(indexMap: Map[Int, Double], input: Array[String]): Map[Int, Double] = {
    val nodes = input.map(item => RankNode(item))
    var count = 0;
    var scoreMap = indexMap
    val scores = Range(0, input.length)
    while (count < params.lmIterations) {

      scores.foreach(index => {
        val crrNode = nodes(index)

        for (k <- 0 until index) {
          val previousNode = nodes(k)
          val d = partition(math.abs(index - k))
          val pairs = previousNode.linkPairs(crrNode)
          val forwardScores = pairs.map { case (src, dst) => {
            val srcId = src.hashCode
            val dstId = dst.hashCode
            val key = dst + skip + d
            val keyId = key.hashCode
            if (states.contains(srcId) && states.contains(dstId)) {
              states(srcId).forwardLog(keyId)
            }
            else {
              0
            }
          }
          }

          val forwardScore = forwardScores.sum / forwardScores.length
          scoreMap = scoreMap + (index -> (0.85f * scoreMap(index) + 0.15f * forwardScore));
        }

        /*for (k <- index + 1 until scores.length) {
          val nextNode = nodes(k)
          val d = partition(math.abs(index - k))
          val pairs = crrNode.linkPairs(nextNode)
          val forwardScores = pairs.par.map { case (src, dst) => {
            val key = src + skipBack + d
            val keyId = key.hashCode
            if (states.contains(src) && states.contains(dst)) {
              states(dst).backwardLog(keyId)
            }
            else {
              0
            }
          }
          }

          val backwardScore = forwardScores.sum / forwardScores.length
          scoreMap = scoreMap + (index -> (0.85f * scoreMap(index) + 0.15f * backwardScore));
        }*/

      })

      count += 1
    }

    scoreMap
  }

  def combinatoric(input: Array[Array[(String, Double)]],
                   result: Array[Array[(String, Double)]] = Array[Array[(String, Double)]](Array()),
                   i: Int = 0): Array[Array[(String, Double)]] = {

    if (i == input.length) result
    else {

      var crr = i;
      val dist = input(crr).distinct

      var array = Array[Array[(String, Double)]]()
      for (k <- 0 until dist.length) {
        for (j <- 0 until result.length) {
          val current = result(j) :+ dist(k)
          array = array :+ current
        }
      }

      combinatoric(input, array, crr + 1)
    }
  }



  def rank(sentence: Array[(String, Double)]): Double = {
    val adjust = ("START", 1d) +: sentence :+ ("END", 1d)
    var scoreMap = Range(0, adjust.length)
      .map(index => index->adjust(index)._2).toMap
    val input = adjust.map(_._1)
    scoreMap = rank(scoreMap, input)
    scoreMap.map(_._2).sum
  }

  def inference(sentence: Array[String]): Array[String] = {
    val splits = sentence.map(token => partitioning(token)).filter(_.nonEmpty)
    val partitions = splits.sliding(params.lmWindowLength, 1).map(input => {
      val combinations = combinatoric(input)
      combinations.map(combinationSequence => {
          (combinationSequence, rank(combinationSequence))
        }).sortBy(item => item._2)
        .reverse
        .map(_._1).head
    }).toArray

    var tokens = partitions.head.flatMap(item => item._1.split(lm.transducer.split))
    partitions.tail.foreach(item => {
      tokens = tokens ++ item.last._1.split(lm.transducer.split)
    })

    tokens
  }


  def getModelFilename(): String = modelFilename


  override def splitSentence(sentence: Array[String]): Array[String] = {
    inference(sentence)
  }



  override def save(): AbstractLM = {
    println("Saving model filename: " + getModelFilename())
    val stream = new ObjectOutputStream(new FileOutputStream(getModelFilename()))

    stream.writeInt(states.size)
    states.foreach(pair => {
      pair._2.save(stream)
    })
    stream.close()
    this
  }

  override def exists(): Boolean = new File(getModelFilename()).exists()

  override def load(transducer: Transducer): AbstractLM = {
    println("Model filename " + getModelFilename())
    dictionary.load()
    val stream = new ObjectInputStream(new FileInputStream(getModelFilename()))

    val size = stream.readInt()
    for (i <- 0 until size) {
      val state = RankState(eps.hashCode).load(stream)
      states = states + (state.wordId -> state)
    }
    lm.transducer = transducer
    stream.close()
    this
  }

  override def load(): AbstractLM = {
    println("Model filename " + getModelFilename())
    dictionary.load()
    val stream = new ObjectInputStream(new FileInputStream(getModelFilename()))

    val size = stream.readInt()
    for (i <- 0 until size) {
      val state = new RankState(eps.hashCode).load(stream)
      states = states + (state.wordId -> state)
    }
    stream.close()
    this
  }

  override def trainSentence(sentence: String): AbstractLM = {
    count(sentence)
  }

  override def loadTrain(): AbstractLM = {
    val range = Range(0, params.lmEpocs)
    val nsize = params.lmThreads
    val locale = new Locale("tr")
    val lock = new Object()
    range.sliding(nsize, nsize).foreach(iset => {
      iset.par.foreach(i => {
        println(s"Epoc: ${i}/${params.lmEpocs} for ${params.adapterName}")
        val newLM = new RankLM(params)
          .setLM(lm)
          .setDictionary(dictionary)
        val start = i * params.lmMaxSentence
        Source.fromFile(params.sentencesFile).getLines()
          .filter(sentence => sentence.length < params.lmMaxSentenceLength)
          .zipWithIndex.filter(_._2 >= start).take(params.lmMaxSentence).map(_._1)
          .toArray
          .foreach(line => {
            val sentence = line.toLowerCase(locale).split("[\\s\\p{Punct}]+")
              .filter(item => item.matches("\\p{L}+"))
              .filter(item => item.length < params.lmTokenLength)
              .mkString(" ")
            newLM.count(sentence)
          })

        lock.synchronized{
          println("Merging...")
          merge(newLM)
        }
      })
    })

    normalize()
      .prune()
      .save()

    this
  }

  override def train(sequence: Array[String]): AbstractLM = ???

  override def train(sequence: String): AbstractLM = ???

  override def trainDictionary(item: String): AbstractLM = ???

  override def trainDictionary(item: Array[String]): AbstractLM = ???



  override def normalize(): AbstractLM = {
    println("Normalizing...")
    states.par.map(pair => pair._2.normalize())
    this
  }

  override def prune(): AbstractLM = {
    println("Prunning")
    states.par.foreach(pair => pair._2.prune(params.lmPrune))
    this
  }


  override def copy(): AbstractLM = {
    val rankLM = new RankLM(params)
    rankLM.lm = lm.copy()
    rankLM.states = states.toArray.map(pair => pair._1 -> pair._2.copy()).toMap
    rankLM.setDictionary(dictionary)
    rankLM
  }

  override def merge(abstractLM: AbstractLM): AbstractLM = {
    val otherLM = abstractLM.asInstanceOf[RankLM]
    otherLM.states.foreach(pair => {
      if (states.contains(pair._1)) {
        states(pair._1).merge(pair._2)
      }
      else {
        states = states + (pair._1 -> pair._2)
      }
    })
    this
  }
}
