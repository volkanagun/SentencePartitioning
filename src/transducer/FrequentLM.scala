package transducer

import experiments.Params
import utils.Tokenizer

class FrequentLM(params: Params) extends AbstractLM(params) {

  val tokenizer = new Tokenizer(windowSize = 2)

  override def copy(): AbstractLM = this

  override def save(): AbstractLM = {
    tokenizer.saveZip()
    this
  }

  override def exists(): Boolean = {
    tokenizer.exists()
  }

  override def load(transducer: Transducer): AbstractLM = {
    tokenizer.loadZip()
    this
  }

  override def load(): AbstractLM = {
    tokenizer.loadZip()
    this
  }

  override def trainSentence(sentence: String): AbstractLM = {
    tokenizer.freqConstruct(sentence)
    this
  }

  override def loadTrain(): AbstractLM = {
    Tokenizer.freqStemConstruct()
    tokenizer.loadZip()
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

  override def findMinSplit(token: String): Array[String] = {
    tokenizer.ngramFilter(token)
  }

  override def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    tokenizer.ngramFilter(sentence)
  }

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = {
    tokenizer.ngramFilter(sentence)
  }



  override def findMultiSplitSentence(sentence: Array[String]): Array[String] = tokenizer.ngramFilter(sentence)

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = tokenizer.ngramFilter(sentence)

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = tokenizer.ngramFilter(sentence)

  override def normalize(): AbstractLM = this

  override def prune(): AbstractLM = this

  override def subsequence(sentence: Array[String]): String = {
    tokenizer.ngramFilter(sentence).mkString(" ")
  }
}
