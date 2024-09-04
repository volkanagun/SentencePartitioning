package transducer

import experiments.Params
import tagging.lemmatizer.{RegexTokenizer, WordLemmatizer, WordSpan}
import utils.Tokenizer.locale

class LemmaLM(params:Params)  extends RankLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/lemma${params.lmID()}.bin"

  val wordLemmatizer = WordLemmatizer.load()
  val regexTokenizer = new RegexTokenizer()

  override def subsequence(sentence: Array[String]): String = {
    findMinSplitSentence(sentence)
      .map(item => item.split(lm.transducer.split)
        .filter(_.nonEmpty)).map(_.mkString(" "))
      .mkString("[",",","]")
  }

  def partition(sentence:Array[String]):Array[Array[String]]={
    var spanList = Array[WordSpan]()
    var start = 0
    for(i<-0 until sentence.length){
      spanList = spanList :+ new WordSpan(start, start+sentence(i).length)
        .setValue(sentence(i))
      start = start + sentence(i).length + 1
    }
    val text = sentence.mkString(" ")
    val lemmaList = wordLemmatizer.lemmatize(spanList, text)
    val lemmaString = lemmaList.map(wordGroup=> wordGroup.lemmaSplitList(lm.transducer.marker).distinct)
    lemmaString
  }

  override def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    lm.pageRank(sentence,  partition, params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = {
    lm.pageRank(sentence, partition, params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = {
    lm.pageSlideRank(sentence,  partition, params.lmSlideLength,params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = findMultiSplitSentence(sentence)

  override def train(sequence: Array[String]): LemmaLM = {
    val f:(Array[String]=> Array[Array[String]]) = partition
    lm.countCombinatoric(sequence,f, params.lmSlideLength, params.lmSkip)
    this
  }

  override def train(sequence: String): LemmaLM = {
    val tokens = regexTokenizer.partition(sequence)
    train(tokens)
  }

  override def trainSentence(line: String): AbstractLM = {
    val arr = regexTokenizer.partition(line)
      .map(_.toLowerCase(locale))
      .sliding(params.lmWindowLength, 1)
      .toArray

    arr.foreach(sequence => train(sequence))
    this
  }

  override def getModelFilename():String={
    modelFilename
  }

  override def loadTrain(): this.type = {
    if (exists()) {
      lm = TransducerOp.loadLM(getModelFilename(), lm.transducer)
      if (params.lmForceTrain || lm.isEmpty()) {
        lm = TransducerOp.trainParallelCombinatoricBySkip(lm,partition,  textFilename, getModelFilename(), params)

        TransducerOp.saveLM(getModelFilename(), lm)
      }
    }
    else {
      lm = TransducerOp.trainParallelCombinatoricBySkip(lm.transducer, textFilename,getModelFilename(), params)
      TransducerOp.saveLM(getModelFilename(), lm)
    }

    this
  }
}


object LemmaLM {

  var lemmaLM: LemmaLM = null

  def apply(windowSize: Int, slideLength: Int): LemmaLM = {
    if (lemmaLM == null) {
      val params = new Params()
      params.lmWindowLength = windowSize
      params.lmSlideLength = slideLength

      lemmaLM = new LemmaLM(params)
      lemmaLM.load()
    }

    lemmaLM
  }

  def trainBig(): Unit = {
    val params = new Params()
    params.lmMaxSentence = 5000
    params.lmEpocs = 1000
    params.lmTrainDictionary = true
    params.lmWindowLength = 5
    params.lmSlideLength = 5
    params.lmSkip = 5

    lemmaLM = new LemmaLM(params)
    lemmaLM.initialize()
      .loadTrain().test()
  }

  def testSmall(): Unit = {
    val params = new Params()
    params.lmTrainDictionary = true
    params.lmMaxSentence = 1000
    params.lmEpocs = 24
    params.lmTopSplit = 2400

    val ngramLM = new LemmaLM(params).loadTrain()

    val array = Array("dolayi zaman", "yasa gerektirdigi", "yasalar simdi", "yasada zaten", "yasaya gore", "yasada var", "yasam yok", "yasan yok", "yasadakiler ne diyor", "yasaninkiler alinti",
      "yasa cok genis")

    val stemArray = array.flatMap(sequence => sequence.split("\\s").flatMap(token => TransducerOp.stemPartition(token)))
    val seqArray = array.map(sequence => sequence)

    stemArray.foreach(sequence => ngramLM.trainDictionary(sequence))
    seqArray.foreach(sequence => ngramLM.train(sequence))
    ngramLM.findMinSplit("yasada var").foreach(item => println(item))

  }

  def testSentence(): Unit = {

    val params = new Params()
    params.lmTrainDictionary = true
    params.lmMaxSentence = 1000
    params.lmEpocs = 24

    val sequence = Array("Yaşamdan", "ve", "sanattan", "anlamayanların", "topluluğudur",".")
    val testSequence = Array("yaşam", "sanatı")
    lemmaLM = new LemmaLM(params).initialize()


    lemmaLM.trainDictionary(sequence)
    lemmaLM.train(testSequence)
    lemmaLM.prune().normalize()

    val subToken = lemmaLM.subsequence(testSequence)
    println(subToken)

  }

  def train(): Unit = {
    val params = new Params()
    params.lmMaxSentence = 1000
    params.lmEpocs = 2
    params.lmTrainDictionary = true
    lemmaLM = new LemmaLM(params)
    lemmaLM.loadTrain()
  }

  def main(args: Array[String]): Unit = {
    testSentence()
  }
}