package transducer

import experiments.Params

class WordLM(params:Params) extends AbstractLM(params) {

  val modelFilename = s"${parent}/resources/transducers/word${params.lmID()}.bin"

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

  override def splitSentence(sentence: Array[String]): Array[String] = sentence

  override def normalize(): AbstractLM = this

  override def prune(): AbstractLM = this

  override def splitToken(token: String): Array[String] = Array(token)
}
