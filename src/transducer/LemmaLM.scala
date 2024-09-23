package transducer

import experiments.Params
import tagging.lemmatizer.{RegexTokenizer, WordLemmatizer, WordSpan}
import utils.Tokenizer.locale

import java.io.File

class LemmaLM(params:Params) extends SkipLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/lemma${params.lmID()}.bin"

  val wordLemmatizer = WordLemmatizer.load()
  val regexTokenizer = new RegexTokenizer()


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
  override def splitSentence(sentence: Array[String]): Array[String] = {
    lm.pageSlideRank(sentence,  partition, params.lmSlideLength,params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def splitToken(token: String): Array[String] = {
    wordLemmatizer.extract(token).flatMap(wordGroup=>{
      wordGroup.lemmaSplitList(lm.transducer.marker)
    })
  }

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

  override def exists(): Boolean = {
    new File(getModelFilename()).exists()
  }

  override def load(): this.type = {
    load(lm.transducer)
    this
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
