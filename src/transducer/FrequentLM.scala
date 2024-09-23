package transducer

import experiments.Params
import utils.Tokenizer

class FrequentLM(params: Params) extends AbstractLM(params) {

  val tokenizer = new Tokenizer(windowSize = 2)
  val tokenizerFilename = params.dictionaryFilename()
  val tokenizerZipFilename = params.dictionaryZipFilename()

  override def copy(): AbstractLM = this

  override def save(): AbstractLM = {
    tokenizer.saveZip(tokenizerZipFilename)
    this
  }

  override def exists(): Boolean = {
    tokenizer.exists(tokenizerZipFilename)
  }

  override def load(transducer: Transducer): AbstractLM = {
    tokenizer.loadZip(tokenizerZipFilename)
    this
  }

  override def load(): AbstractLM = {
    tokenizer.loadZip(tokenizerZipFilename)
    this
  }

  override def trainSentence(sentence: String): AbstractLM = {
    tokenizer.freqConstruct(sentence)
    this
  }

  override def loadTrain(): AbstractLM = {

    Tokenizer.freqStemConstruct(params.sentencesFile, params, tokenizerZipFilename)
    tokenizer.loadZip(tokenizerZipFilename)
    this
  }

  override def train(sequence: Array[String]): AbstractLM = {
    this
  }

  override def train(sequence: String): AbstractLM = {
    tokenizer.freqConstruct(sequence)
    this
  }

  override def trainDictionary(item: String): AbstractLM = this

  override def trainDictionary(item: Array[String]): AbstractLM = this


  override def splitToken(token: String): Array[String] = {
    tokenizer.ngramTokenFilter(token)
  }

  override def splitSentence(sentence: Array[String]): Array[String] = tokenizer.ngramFilter(sentence)

  override def normalize(): AbstractLM = this

  override def prune(): AbstractLM = this

}
