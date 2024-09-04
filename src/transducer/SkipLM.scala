package transducer

import experiments.Params
import utils.Tokenizer.locale

import java.io.File

class SkipLM(override val params: Params) extends AbstractLM(params) {


  val modelFilename = s"${parent}/resources/transducers/skip${params.lmID()}.bin"

  override def subsequence(sentence: Array[String]): String = {
    findMinSplitSentence(sentence)
      .flatMap(item => item.split(lm.transducer.split)
        .filter(_.nonEmpty))
      .mkString(" ")
  }

  override def normalize(): AbstractLM = {
    lm.normalize()
    this
  }


  override def prune(): AbstractLM = {
    lm.prune(params.lmPrune)
    this
  }

  def train(sequence: Array[String]): SkipLM = {
    lm.countCombinatoric(sequence, params.lmSlideLength, params.lmSkip)
    this
  }

  def train(sequence: String): SkipLM = {
    lm.countCombinatoric(sequence.toCharArray.map(_.toString), params.lmSlideLength, params.lmSkip)
    this
  }

  def trainDictionary(item: String): SkipLM = {
    lm.transducer.addPrefix(item)
    this
  }

  def trainDictionary(item: Array[String]): SkipLM = {
    lm.transducer.addPrefixes(item)
    this
  }

  def findMinSplit(token: String): Array[String] = {
    lm.inferMinSplit(token, params.lmTopSplit)
  }

  def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    lm.skipLink(sentence, params.lmSkip, params.lmTopSplit)
  }

  def findMinSplitEfficient(sentence: Array[String]): Array[String] = {
    lm.skipEfficientLink(sentence, params.lmSkip, params.lmTopSplit)
  }

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = {
    lm.skipSlideLink(sentence, params.lmWindowLength, params.lmSkip, params.lmTopSplit)
  }

  override def findMultiSplitSentence(sentence: Array[String]): Array[String] = {
    lm.skipLink(sentence, params.lmSkip, params.lmTopSplit)
  }

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = {
    lm.skipLink(sentence, params.lmSkip, params.lmTopSplit)
  }

  def getModelFilename():String={
    modelFilename
  }


  override def exists(): Boolean = new File(getModelFilename()).exists()

  override def load(transducer: Transducer): SkipLM = {
    if (exists()) {
      println("Loading filename: " + getModelFilename())
      lm = TransducerOp.loadLM(getModelFilename(), transducer)
    }
    else {
      println(s"SkipLM model: ${getModelFilename()} not found")
      lm = new TransducerLM(transducer)
    }

    this
  }

  override def load(): SkipLM = {
    load(lm.transducer)
    this
  }


  override def trainSentence(line: String): AbstractLM = {
    val arr = line.split("\\s+")
      .map(_.toLowerCase(locale))
      .filter(item => item.matches("\\p{L}+"))
      .sliding(params.windowLength, 1)
      .toArray

    arr.foreach(sequence => lm.countCombinatoric(sequence, params.lmSlideLength, params.lmSkip))
    this
  }

  override def loadTrain(): this.type = {
    if (exists()) {

      lm = TransducerOp.loadLM(modelFilename, lm.transducer)
      if(params.lmForceTrain || lm.isEmpty()) {

        lm = TransducerOp.trainParallelCombinatoricBySkip(lm, textFilename, modelFilename, params)
        TransducerOp.saveLM(modelFilename, lm)
      }
    }
    else {
      lm = TransducerOp.trainParallelCombinatoricBySkip(lm.transducer, textFilename, modelFilename, params)
      TransducerOp.saveLM(modelFilename, lm)
    }

    this
  }

  override def copy(): AbstractLM = {
    val skipLM = new SkipLM(params)
    skipLM.lm = lm.copy()
    this
  }

  override def save(): AbstractLM = {
    TransducerOp.saveLM(modelFilename, lm)
    this
  }
}

object SkipLM {

  //Test it here
  def apply(windowSize:Int, maxSkip:Int, topSplit:Int) : SkipLM={
    val params = new Params()
    params.lmSlideLength = 4
    params.lmWindowLength = windowSize
    params.lmSkip = maxSkip
    params.lmTopSplit = topSplit
    val skipLM = new SkipLM(params).load()
    //skipLM.test()
    skipLM
  }

  def trainBig(): Unit = {
    val lmParams = new Params()
    lmParams.lmSkip = 3
    lmParams.lmSlideLength = 7
    val rankingLM = new SkipLM(lmParams)
  }

  def test1(): Unit = {
    val params = new Params()
    params.lmSlideLength = 4
    params.lmWindowLength = 5
    params.lmSkip = 3
    val rankingLM = new SkipLM(params)
    rankingLM.train(Array("hast", "ane", "ler", "de"))
    rankingLM.trainDictionary(Array("hast", "ane", "ler", "de"))
    rankingLM.normalize()
    val array1 = "hastanelerde".toCharArray.map(_.toString)
    val result1 = rankingLM.findMinSplitSentence(array1)
    println("Sentence: " + array1.mkString(" ") + "\n" + result1.mkString(" "))
  }

  def test2(): Unit = {
    val params =  new Params()
    params.lmSlideLength = 4
    params.lmWindowLength = 5
    params.lmSkip = 3
    params.lmTrainDictionary = false

    val rankingLM = new SkipLM(params)

    rankingLM.train(Array("hastane", "lerde", "adam", "yok"))
    rankingLM.train(Array("ev", "lerde", "adam", "yok"))
    rankingLM.train(Array("her", "yerde", "adam", "çok"))
    rankingLM.train(Array("zaman", "yok", "adam", "çok"))
    rankingLM.trainDictionary(Array("zaman", "çok", "adam", "yok", "hastane", "ev", "her", "yerde", "lerde"))

    val array1 = "zaman çok".toCharArray.map(_.toString)
    val result1 = rankingLM.findMinSplitSentence(array1)
    println("Sentence: " + array1.mkString(" ") + "\n" + result1.mkString(" "))
  }

  def test3(): Unit = {
    val params = new Params()
    params.lmTrainDictionary = true
    params.lmEpocs = 10
    params.lmMaxSentence = 1000
    val rankingLM = new SkipLM(params).loadTrain()
    val array1 = "zaman çok".toCharArray.map(_.toString)
    val result1 = rankingLM.findMinSplitSentence(array1)
    println("Sentence: " + array1.mkString(" ") + "\n" + result1.mkString(" "))
  }

  def main(args: Array[String]): Unit = {
    test3()
  }
}
