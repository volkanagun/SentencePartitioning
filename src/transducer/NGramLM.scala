package transducer

import experiments.Params
import utils.Tokenizer

import java.io.File


class NGramLM(override val params: Params) extends AbstractLM(params) {

  val modelFilename = s"${parent}/resources/transducers/ngrams${params.lmID()}.bin"

  override def subsequence(sentence: Array[String]): String = {
    findLikelihoodSentence(sentence)
      .flatMap(item => item.split(lm.transducer.split)
        .filter(_.nonEmpty))
      .mkString(" ")
  }

  override def copy(): NGramLM = {
    val ngramLM = new NGramLM(params)
    ngramLM.lm = lm.copy()
    ngramLM
  }

  override def normalize(): AbstractLM = {
    lm.normalize()
    this
  }


  override def prune(): AbstractLM = {
    lm.prune(params.lmPrune)
    this
  }


  override def save(): NGramLM.this.type = {

    TransducerOp.saveLM(modelFilename, lm)

    this
  }

  override def exists(): Boolean = {
    new File(modelFilename).exists()
  }

  override def load(input: Transducer): NGramLM = {

    if (exists()) {
      lm = TransducerOp.loadLM(modelFilename, input)

    }
    else {
      lm = new TransducerLM(input)

    }

    this
  }

  override def load(): NGramLM = {
    load(new Transducer())
    this
  }


  override def trainSentence(sentence: String): NGramLM = {
    val arr = sentence.split("\\s+")
      .map(_.toLowerCase(Tokenizer.locale))
      .filter(item => item.matches("\\p{L}+"))
      .sliding(params.lmWindowLength, 1)
      .toArray

    arr.foreach(sequence => lm.countCombinatoric(sequence, params.lmTopSplit, params.lmSlideLength))
    this
  }

  override def loadTrain(): NGramLM = {

    if (exists()) {

      lm = TransducerOp.loadLM(modelFilename, new Transducer())

      if(params.lmForceTrain || lm.isEmpty()) {
        lm = TransducerOp.trainParallelCombinatoricBySlide(lm, textFilename, modelFilename, params)

        TransducerOp.saveLM(modelFilename, lm)
      }

    }
    else {
      lm = TransducerOp.trainParallelCombinatoricBySlide(new Transducer(), textFilename, modelFilename, params)
      TransducerOp.saveLM(modelFilename, lm)
    }

    this
  }

  override def train(sequence: Array[String]): NGramLM = {
    lm.countCombinatoric(sequence, params.lmTopSplit, params.lmSlideLength)
    this
  }

  override def train(sequence: String): NGramLM = {
    lm.count(sequence, params.lmTopSplit, params.lmSlideLength)
    this
  }

  override def trainDictionary(item: String): NGramLM = {
    lm.transducer.addPrefix(item)
    this
  }

  override def trainDictionary(item: Array[String]): NGramLM = {
    lm.transducer.addPrefixes(item)
    this
  }

  override def findMinSplit(sequence: String): Array[String] = {
    lm.inferMinSplit(sequence, params.lmTopSplit)
  }


  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = {
    lm.inferMinTokenSplit(sentence, params.lmTopSplit)
  }

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = lm.inferSlideMinTokenSplit(sentence, params.lmSlideLength, params.lmTopSplit)

  override def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    lm.inferMinTokenSplit(sentence, params.lmTopSplit)
  }

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = {
    lm.inferMinTokenSplit(sentence, params.lmTopSplit)
  }

  override def findMultiSplitSentence(sentence: Array[String]): Array[String] = {
    lm.inferMultiTokenSplit(sentence, params.lmTopSplit)
  }

}

object NGramLM {

  var ngramLM: NGramLM = null

  def apply(windowSize: Int, slideLength: Int, topSplit: Int): NGramLM = {
    if (ngramLM == null) {
      val params = new Params()
      params.lmWindowLength = windowSize
      params.lmSlideLength = slideLength
      params.lmTopSplit = topSplit

      ngramLM = new NGramLM(params)
      ngramLM.load()
    }

    ngramLM
  }

  def trainBig(): Unit = {
    val params = new Params()
    params.lmMaxSentence = 500
    params.lmEpocs = 1
    params.lmTrainDictionary = true
    params.lmWindowLength = 2
    params.lmSlideLength = 7

    ngramLM = new NGramLM(params)
    ngramLM.initialize()
      .loadTrain().test()

  }

  def testSmall(): Unit = {
    val params = new Params()
    val ngramLM = new NGramLM(params)
    val slide = 3;
    val array = Array("dolayi zaman", "yasa gerektirdigi", "yasalar simdi", "yasada zaten", "yasaya gore", "yasada var", "yasam yok", "yasan yok", "yasadakiler ne diyor", "yasaninkiler alinti",
      "yasa cok genis")

    val stemArray = array.flatMap(sequence => sequence.split("\\s").flatMap(token => TransducerOp.stemPartition(token)))
    val seqArray = array.map(sequence => sequence)

    stemArray.foreach(sequence => ngramLM.trainDictionary(sequence))
    seqArray.foreach(sequence => ngramLM.train(sequence))
    ngramLM.findMinSplit("görece var").foreach(item => println(item))

  }

  def testSentence(): Unit = {

    val params =  new Params()
    params.lmTrainDictionary = true

    val sequence = Array("yaşamdan", "sanat", "yaşam", "ağacı", "yaşam", "yaşamaya", "yaşamda", "yaşam", "sanata", "yaşamdan", "ağaç", "ağaçları", "sanatları")
    val testSequence = Array("yaşam", "sanatı")
    ngramLM = new NGramLM(params).initialize()

    ngramLM.trainDictionary(sequence)
    ngramLM.train(testSequence).normalize()
    val subToken = ngramLM.subsequence(testSequence)
    println(subToken)

  }

  def train(): Unit = {
    val params = new Params()
    params.lmMaxSentence = 1000
    params.lmEpocs = 2
    params.lmTrainDictionary = true
    ngramLM = new NGramLM(params)
    ngramLM.loadTrain()
  }

  def main(args: Array[String]): Unit = {
    //train()
    trainBig()

    //testSmall()
    testSentence()
  }
}