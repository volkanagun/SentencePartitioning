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
  def splitSpace = s"(${lm.transducer.split}|\\s+)"

  def initialize(): this.type = {
    if(!exists()) {
      lm.transducer = TransducerOp.fromDictionary(lm.transducer, dictionaryFilename, dictionaryTextFilename, params)
      lm.transducer = TransducerOp.fromText(lm.transducer, textFilename, params)
      lm = new TransducerLM(lm.transducer)
      save()
      System.gc()
      this
    }
    else{
      load()
      this
    }
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

  def findMinSplit(token: String): Array[String]

  def findMinSplitSentence(sentence: Array[String]): Array[String]
  def findMinSplitEfficient(sentence: Array[String]): Array[String]
  def findMinSlideSplitSentence(sentence:Array[String]):Array[String]
  def findMultiSplitSentence(sentence: Array[String]): Array[String]

  def findLikelihoodSentence(sentence: Array[String]): Array[String]
  def normalize():AbstractLM
  def prune():AbstractLM


  def infer(token: String): Array[String] = {
    lm.infer(token, params.lmTopSplit)
  }

  def inferMulti(token: String): Array[String] = {
    val result = lm.infer(token, params.lmTopSplit)
    val splitted = result.flatMap(token=> token.split(lm.transducer.split))
    result ++ splitted
  }

  def inferWindow(sentence:String):Array[String] = {
    val sequence = wordTokenizer.standardTokenizer(sentence)
    sequence.sliding(params.lmWindowLength, params.lmWindowLength).map(window=> {
      findMinSplitSentence(window).mkString(" ")
    }).toArray
  }

  def inferWindowMulti(sentence:String):Array[String] = {
    val sequence = wordTokenizer.standardTokenizer(sentence)
    val samples = sequence.sliding(params.lmWindowLength, params.lmWindowLength).map(window=> {
      findMinSplitSentence(window).mkString(" ")
    }).toArray
    val combined = samples.map(sentence=> sentence.split(lm.transducer.split).map(_.trim).mkString(" "))
    samples ++ combined
  }





  def subsequence(sentence:Array[String]):String

  def test(): Unit = {
    lm.test()
  }
}
