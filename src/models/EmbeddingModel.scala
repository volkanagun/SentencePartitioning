package models

import evaluation.IntrinsicFunction
import org.deeplearning4j.nn.graph.ComputationGraph
import utils.{Params, Tokenizer}

abstract class EmbeddingModel(val params: Params) extends IntrinsicFunction {

  var avgTime = 0d
  var sampleCount = 0

  var dictionaryIndex = Map[String, Int]("dummy" -> 0)
  var dictionary = Map[String, Array[Float]]()
  var computationGraph:ComputationGraph = null

  lazy val tokenizer = new Tokenizer().load()

  def getTrainTime():Double = avgTime

  def train(filename: String): EmbeddingModel

  def save(): EmbeddingModel

  def load(): EmbeddingModel


  def getDictionary():Map[String, Array[Float]] = dictionary
  def getDictionaryIndex():Map[Int, Array[Float]]={
    dictionary.map{case(ngram, vector)=> dictionaryIndex(ngram)->vector}
  }

  def update(ngram:String, vector:Array[Float]):Int={
    dictionary = dictionary.updated(ngram, vector)
    update(ngram)
  }

  def update(ngram: String): Int = {
    if (dictionaryIndex.size < params.dictionarySize) {
      dictionaryIndex = dictionaryIndex.updated(ngram, dictionaryIndex.getOrElse(ngram, dictionaryIndex.size))
    }

    retrieve(ngram)
  }

  def retrieve(ngram: String): Int = {
    dictionaryIndex.getOrElse(ngram, 0)
  }

  def tokenize(sentence: String): Array[String] = {
    tokenizer.ngramFilter(sentence)
  }

  def forward(token: String): Array[Float] = {
    val frequentNgrams = tokenizer.ngramStemFilter(token).filter(ngram => dictionary.contains(ngram))
    val ngramVectors = frequentNgrams.map(ngram => dictionary(ngram))
    average(ngramVectors)
  }

  def average(embeddings: Array[Array[Float]]): Array[Float] = {
    var foldResult = Array.fill[Float](params.embeddingLength)(0f)
    embeddings.foldRight[Array[Float]](foldResult) { case (a, main) => {
      main.zip(a).map(pair => pair._1 + pair._2)
    }
    }
  }
}
