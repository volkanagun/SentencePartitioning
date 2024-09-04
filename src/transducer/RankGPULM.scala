package transducer

import experiments.Params

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source

class RankGPULM(params: Params) extends SkipLM(params) {


  override val modelFilename = s"${parent}/resources/transducers/rank-gpu${params.lmID()}.bin"

  var maxDictionary = 1000000
  var maxWindow = 50
  val dummyIndex = 0
  var dictionary = Map[String, Int]("dummy" -> dummyIndex)

  var counts: Map[RankGpuKey, Float] = Map()


  override def getModelFilename(): String = modelFilename

  def setMaxDictionary(maxDictionary: Int): this.type = {
    this.maxDictionary = maxDictionary
    this
  }

  def setCounts(counts: Map[RankGpuKey, Float]): this.type = {
    this.counts = counts
    this
  }

  def setDictionary(dictionary: Map[String, Int]): this.type = {
    this.dictionary = dictionary
    this
  }

  override def initialize(): this.type = {
    if (!exists()) {
      lm.transducer = TransducerOp.fromDictionary(lm.transducer, dictionaryFilename, dictionaryTextFilename, params)
      lm.transducer = TransducerOp.fromText(lm.transducer, textFilename, params)
      lm = new TransducerLM(lm.transducer)

      save()
      System.gc()
      this
    }
    else {
      load()
      this
    }
  }


  def contains(item: String): Boolean = {
    dictionary.contains(item)
  }


  def index(item: String): Int = {
    synchronized {
      if (dictionary.contains(item)) dictionary(item)
      else if (dictionary.size >= maxDictionary) 0
      else {
        val index = dictionary.size
        dictionary = dictionary.updated(item, dictionary.getOrElse(item, index))
        index
      }
    }
  }

  def index(items: Array[String]): Array[Int] = {
    items.map(index)
  }

  def convert(items: Array[Array[Int]]): Map[Int, Int] = {
    var map = Map[Int, Int]()
    items.foreach(seq => map = convert(seq, map))
    map
  }

  def count(items: Array[Array[Array[Int]]], map: Map[Int, Int], countClone: Map[RankGpuKey, Float]): Map[RankGpuKey, Float] = {
    var clone = countClone
    map.foreach { case (i, ii) => {
      map.foreach { case (j, jj) => {
        for (d <- 0 until maxWindow) {
          val count = items(ii)(jj)(d)
          val key = new RankGpuKey(i, j, d)
          clone = clone.updated(key, clone.getOrElse(key, 0f) + count)
        }
      }
      }
    }
    }

    clone
  }

  def convert(items: Array[Int], map: Map[Int, Int] = Map()): Map[Int, Int] = {
    var nmap = map
    items.foreach(i => {
      nmap = nmap.updated(i, nmap.getOrElse(i, map.size))
    })
    nmap
  }

  override def copy(): AbstractLM = {
    new RankGPULM(params)
      .setDictionary(dictionary)
      .setCounts(counts)
      .setMaxDictionary(maxDictionary)
  }

  override def save(): AbstractLM = {
    val stream = new ObjectOutputStream(new FileOutputStream(modelFilename))
    lm.save(stream)

    stream.writeInt(maxDictionary)
    stream.writeInt(maxWindow)
    stream.writeObject(counts)
    stream.writeObject(dictionary)
    stream.close()
    this
  }

  override def exists(): Boolean = new File(modelFilename).exists()

  override def load(transducer: Transducer): RankGPULM = {
    load()
  }

  override def load(): RankGPULM = {
    if (exists()) {
      val stream = new ObjectInputStream(new FileInputStream(modelFilename))
      lm.load(stream)
      maxDictionary = stream.readInt()
      maxWindow = stream.readInt()
      counts = stream.readObject().asInstanceOf[Map[RankGpuKey, Float]]
      dictionary = stream.readObject().asInstanceOf[Map[String, Int]]
      println("Loading finished...")
    }
    this
  }

  override def trainSentence(sentence: String): AbstractLM = this

  override def loadTrain(): this.type = {
    val range = Range(0, params.lmEpocs)

    range.sliding(params.lmMaxSentence, params.lmMaxSentence).foreach(sequence => {
      sequence.toArray.foreach(epoc => {
        println("Processing epoc: " + epoc)
        val startIndex = epoc * params.lmMaxSentence
        Source.fromFile(textFilename).getLines()
          .filter(sentence => sentence.length < params.maxSentenceLength)
          .zipWithIndex.filter(pair => pair._2 >= startIndex)
          .take(params.lmMaxSentence)
          .map(pair => pair._1)
          .flatMap(sentence => {
            wordTokenizer.standardTokenizer(sentence).sliding(params.lmWindowLength, 1).toArray
              .par.map(windowSeq => {
                gpuSymbols(windowSeq)
              }).map(pair => gpuCount(pair._1, pair._2))
          }).toArray.foreach(append)

      })

      System.gc()
    })

    average()
    this
  }

  def append(countMap: Map[RankGpuKey, Float]): Unit = {
    countMap.foreach { case (key, cnt) => counts = counts.updated(key, counts.getOrElse(key, 0f) + cnt) }
  }

  def padding(sequence: Array[Int]): Array[Int] = {
    Array.fill[Int](maxWindow - sequence.length)(dummyIndex) ++ sequence
  }

  def average(): Unit = {
    val sumMap = counts.keys.map(key => {
      (key.i -> counts.filter(pair => pair._1.i == key.i).map(_._2).sum)
    }).toMap

    counts = counts.map { case (key, count) => (key -> count / sumMap(key.i)) }
  }

  def gpuSymbols(sequence: Array[String]): (Array[Array[Int]], Map[Int, Int]) = {
    val splitting = sequence.map(token => lm.transducer.multipleSplitSearch(token, params.lmTopSplit))
    val combinatorics = lm.combinatoric(splitting).flatMap(tokens => tokens.map(token => token.split(lm.transducer.split)))
    val symbols = combinatorics.map(seq => index(seq))
    val map = convert(symbols)
    val symbolConverted = symbols.map(seq => seq.map(map(_))).map(seq => padding(seq))
    (symbolConverted, map)
  }

  def gpuCount(sequence: Array[Array[Int]], map: Map[Int, Int], mapCount: Map[RankGpuKey, Float] = Map()): Map[RankGpuKey, Float] = {
    val items = LMCountKernel.execute(sequence, map.size, maxWindow)
    count(items, map, mapCount)
  }


  def score(isym: Int, jsym: Int, dd: Int): Float = {
    val key = new RankGpuKey(isym, jsym, dd)
    counts.getOrElse(key, 0f)
  }

  override def findMinSplit(token: String): Array[String] = lm.transducer.multipleSplitSearch(token, 1)
    .head
    .split(lm.transducer.split)


  def score(sequence: Array[Int]): Double = {

    var scoreMap = Map[Int, Double](0 -> 1d)
    var count = 0
    var range = Range(0, sequence.length)
    while (count < params.lmIterations) {

      range.foreach(i => {
        val isym = sequence(i)
        val sum = score(isym, isym, 0)
        var jsum = 0.85f * scoreMap(i) + 0.15f * sum
        val min = Math.max(0, i - params.lmWindowLength)
        for (k <- min until i) {
          val ksym = sequence(k)
          val d = i - k
          val sumSkip = score(isym, ksym, d)
          jsum += 0.85f * scoreMap(i) + 0.15f * sumSkip;
        }

        jsum = jsum / params.lmIterations
        scoreMap = scoreMap.updated(i, jsum)
      })

      count += 1
    }

    val norm = Math.pow(sequence.length, 2)
    scoreMap.map(_._2).sum / norm
  }

  override def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    //convert to int use page rank
    val sequence = sentence.map(token => lm.transducer.multipleSplitSearch(token, params.lmTopSplit))
    val combinations = lm.combinatoric(sequence)
    val maxIndex = combinations.map(items => index(items)).zipWithIndex.map(pair => (score(pair._1), pair._2))
      .maxBy(_._1)._2

    combinations(maxIndex)
  }

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = findMinSplitSentence(sentence)

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = {
    findMinSplitSentence(sentence)
  }

  override def findMultiSplitSentence(sentence: Array[String]): Array[String] = findMinSplitSentence(sentence)

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = findMinSplitSentence(sentence)

  override def normalize(): SkipLM = this

  override def prune(): AbstractLM = this

  override def subsequence(sentence: Array[String]): String = sentence.mkString(" ")
}
