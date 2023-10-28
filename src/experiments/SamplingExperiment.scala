package experiments

import evaluation.{ExtrinsicNER, ExtrinsicPOS, ExtrinsicSentiment, IntrinsicEvaluation}
import models.EmbeddingModel
import utils.Params

import java.io.File
import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport

object SamplingExperiment {

  val samplingNames = Array("VocabSelect", "VotedDivergence",  "KMeans", "KL", "VE", "LM", "Mahalonabis", "Euclidean", "Entropy",  "KL", "Least",  "Boltzmann")
  val adapterNames = Array("avg")

  val selectionSizes = Array(1000, 5000, 25000)
  val models = Array("cbow","skip", "lstm")
  val jsonFilename = "resources/evaluations/sentence-tr.json"

  val embedParams = new Params()

  def experimentKey(embedParams: Params): Int = {
    val extractorName = "feature"
    val embeddingSize = 300
    val hiddenSize = 20
    val clusterSize = 20
    val ngramCombinationSize = 10
    val windowSize = 20
    val committee = 10
    val knn = 7
    val tokenLength = 5
    val dictionarySize = 100000
    val secondDictionarySize = 5000

    val array = Array[Int](
      embedParams.maxSelectSize,
      dictionarySize,
      secondDictionarySize,
      embeddingSize,
      hiddenSize,
      windowSize,
      ngramCombinationSize,
      clusterSize,
      tokenLength,
      true.hashCode(),
      committee,
      knn,
      extractorName.hashCode,
      embedParams.selectionMethod.hashCode)

    val keyID = array.foldRight[Int](7) { case (crr, main) => main * 7 + crr }
    keyID
  }

  def evaluate(embedParams: Params, model: String): this.type = {

    val key = experimentKey(embedParams.modelName(model))
    val sentenceFilename = embedParams.textFolder + embedParams.selectionMethod + "-"+ embedParams.maxSelectSize + "-" + key + ".txt"

    val modelling = model + "/" + embedParams.selectionMethod + "/intrinsic/"+embedParams.embeddingModel + "-"+embedParams.maxSelectSize + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Result filename: "+resultName)
      val mainEvaluation = new IntrinsicEvaluation(resultName)
        .attachEvaluations(jsonFilename)
        .compile()

      val extrinsicNER = new ExtrinsicNER(embedParams)
      val extrinsicPOS = new ExtrinsicPOS(embedParams)
      val extrinsicSentiment = new ExtrinsicSentiment(embedParams)

      mainEvaluation.functions :+= extrinsicNER
      mainEvaluation.functions :+= extrinsicPOS
      mainEvaluation.functions :+= extrinsicSentiment

      mainEvaluation.filter(Array("SEMEVAL"))
      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel)
        .train(embedParams.corpusFilename())

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
    }
    else if(new File(resultName).exists()){
      println("Found: "+resultName)
    }
    this
  }


  def evaluate(mainEvaluation: IntrinsicEvaluation, model: EmbeddingModel, params: Params): Unit = {

    println("==============================================================")
    println("==========" + params.embeddingModel.toUpperCase + "===============")
    println("==============================================================")

    mainEvaluation.evaluateReport(model, params)
  }

  def main(args: Array[String]): Unit = {

    System.setProperty("org.bytedeco.openblas.load", "mkl")


      val parCollection = samplingNames.par
      parCollection.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(1))
      parCollection.foreach(scorerName => {
        selectionSizes.foreach(selectSize=>{
          adapterNames.foreach(adapterName=>{
            models.foreach(crrModel => {
              val crrParams = embedParams.copy()
              crrParams.selectionMethod = scorerName
              crrParams.maxSelectSize = selectSize
              crrParams.adapterName = adapterName
              evaluate(crrParams, crrModel)
            })
          })
        })
    })
  }
}
