package transducer

import experiments.Params

import java.io.File

abstract class AbstractLM(val params: Params) extends Serializable{

  val parent = new File(".")
    .getAbsoluteFile()
    .getParentFile()
    .getAbsolutePath
  var dictionaryFilename = s"${parent}/resources/transducers/dictionary.bin"
  var textFilename = s"${parent}/resources/text/sentences/sentences-tr.txt"
  var dictionaryTextFilename = s"${parent}/resources/dictionary/lexicon.txt"


  val wordTokenizer = new WordTokenizer().loadBinary()
  var lm: TransducerLM = new TransducerLM(new Transducer())

  def marker = lm.transducer.marker
  def split = lm.transducer.split
  def getParams = params

  def partition(d: Int): String = {
    if (d >= 5) "dist" else if (d > 2) "neig" else "loc"
  }

  def initialize(): this.type = {

    if(!exists() || lm.isEmpty()) {
      lm.transducer = TransducerOp.fromDictionary(lm.transducer, dictionaryFilename, dictionaryTextFilename, params)
      lm.transducer = TransducerOp.fromText(lm.transducer, textFilename, params)
      lm = new TransducerLM(lm.transducer)
      save()
      System.gc()
    }

    this

  }

  def isEmpty():Boolean={
    lm.isEmpty()
  }

  def graphStats():Map[String, Double]={
    lm.graphStats()
  }

  def setTextFile(textFile:String):this.type ={
    this.textFilename = textFile;
    this
  }

  def copy():AbstractLM

  def save():AbstractLM
  def merge(abstractLM: AbstractLM): AbstractLM = {
    lm.merge(abstractLM.lm)
    this
  }

  def exists(): Boolean

  def load(transducer: Transducer): AbstractLM

  def load(): AbstractLM

  def trainSentence(sentence:String):AbstractLM

  def loadTrain(): AbstractLM

  def train(sequence: Array[String]): AbstractLM

  def train(sequence: String): AbstractLM

  def trainDictionary(item: String): AbstractLM

  def trainDictionary(item: Array[String]): AbstractLM

  def splitSentence(sentence:Array[String]):Array[String]
  def normalize():AbstractLM
  def prune():AbstractLM
  def splitToken(token: String): Array[String]

}
