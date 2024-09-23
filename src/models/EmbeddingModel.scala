package models

import evaluation.IntrinsicFunction
import experiments.Params
import org.deeplearning4j.nn.graph.ComputationGraph
import transducer.{AbstractLM, FrequentLM}
import utils.Tokenizer

import java.io.File
import java.util.Locale
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

abstract class EmbeddingModel(val params: Params, val tokenizer: Tokenizer, val lm: AbstractLM) extends IntrinsicFunction {

  var avgTime = 0d
  var sampleCount = 0
  var locale = new Locale("tr")
  var dictionaryIndex = Map[String, Int]("dummy" -> 0)
  var dictionary = Map[String, Array[Float]]("dummy" -> Array.fill[Float](params.embeddingLength)(0f))
  var computationGraph: ComputationGraph = null
  //lazy val lmFunc = new experiments.LMDataset().partitionFunc(params)

  def getTrainTime(): Double = avgTime

  def train(filename: String): EmbeddingModel

  def save(): EmbeddingModel

  def load(): EmbeddingModel


  def getDictionary(): Map[String, Array[Float]] = dictionary

  def getDictionaryIndex(): Map[Int, Array[Float]] = {
    dictionary.map { case (ngram, vector) => dictionaryIndex(ngram) -> vector }
  }

  def update(ngram: String, vector: Array[Float]): Int = {
    dictionary = dictionary.updated(ngram, vector)
    update(ngram)
  }

  def update(ngram: String): Int = {

    if (dictionaryIndex.size < params.evalDictionarySize) {
      dictionaryIndex = dictionaryIndex.updated(ngram, dictionaryIndex.getOrElse(ngram, dictionaryIndex.size))
    }
    retrieve(ngram)

  }

  def update(ngram: String, size: Int): Int = {

    if (dictionaryIndex.size < size) {
      dictionaryIndex = dictionaryIndex.updated(ngram, dictionaryIndex.getOrElse(ngram, dictionaryIndex.size))
    }

    retrieve(ngram)

  }

  def retrieve(ngram: String): Int = {
    dictionaryIndex.getOrElse(ngram, 0)
  }

  def tokenize(sentence: String): Array[String] = {
    val lwSentence = sentence.toLowerCase(locale)
    val ngrams = tokenizer.ngramFilter(lwSentence)
    val result = ngrams
      .flatMap(ngram => ngram.split("\\s+"))
    result
  }

  /*def tokenize(sentence: String): Array[Array[String]] = {
    lmFunc(sentence)
  }*/


  def forward(token: String): Array[Float] = {
    val lowercase = token.toLowerCase(locale)
    val frequentNgrams = lm.splitToken(lowercase)
      .flatMap(ngram => ngram.split("[\\s+\\#$]"))
      .distinct
      .filter(ngram => dictionary.contains(ngram))

    val ngramVectors = frequentNgrams.map(ngram => dictionary(ngram))
    val result = avg(ngramVectors, params.embeddingLength)
    result
  }

  def split(token:String):Array[String]={
    val lowercase = token.toLowerCase(locale)
    val frequentNgrams = lm.splitToken(lowercase)
      .flatMap(ngram => ngram.split("[\\s+\\#$]"))
      .distinct
      .filter(item =>  item.length >= 3)

    frequentNgrams
  }

}


object EmbeddingModel {


  def clean(): Unit = {
    val modelFolder = "/resources/models/"
    new File("resources/embeddings/").listFiles().par.map(file => {
      (file, new CBOWModel(new Params, new Tokenizer(), new FrequentLM(new Params)).load(file.getAbsolutePath))
    }).filter { case (_, model) => model.dictionaryIndex.exists(pairs => pairs._1.length > 1 && pairs._1.contains("#")) || model.dictionaryIndex.size <= 1 }.foreach(pair=>{
      println("Deleting filename: " + pair._1.getAbsolutePath)
      pair._1.delete()
      val name = pair._1.getName
      val modelName = modelFolder + name.substring(0, name.lastIndexOf(".")) + ".zip"
      new File(modelName).delete()
    })
  }

  def main(args: Array[String]): Unit = {
    clean()
  }
}