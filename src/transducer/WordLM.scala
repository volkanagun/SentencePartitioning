package transducer

import experiments.Params
import utils.Tokenizer

class WordLM(params:Params) extends AbstractLM(params) {


  override def copy(): AbstractLM = this

  override def save(): AbstractLM = this

  override def exists(): Boolean = true

  override def load(transducer: Transducer): AbstractLM = this

  override def load(): AbstractLM = this

  override def trainSentence(sentence: String): AbstractLM = this

  override def loadTrain(): AbstractLM = this

  override def train(sequence: Array[String]): AbstractLM = this

  override def train(sequence: String): AbstractLM = this

  override def trainDictionary(item: String): AbstractLM = this

  override def trainDictionary(item: Array[String]): AbstractLM = this

  override def findMinSplit(token: String): Array[String] = Array(token)

  override def findMinSplitSentence(sentence: Array[String]): Array[String] = sentence

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = sentence

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = sentence

  override def findMultiSplitSentence(sentence: Array[String]): Array[String] = sentence

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = sentence

  override def normalize(): AbstractLM = this

  override def prune(): AbstractLM = this

  override def subsequence(sentence: Array[String]): String = sentence.mkString(" ")
}
