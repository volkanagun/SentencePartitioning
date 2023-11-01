package experiments

import evaluation.{ExtrinsicLSTM, ExtrinsicNER, ExtrinsicPOS, ExtrinsicSentiment, IntrinsicEvaluation}
import models.EmbeddingModel
import sampling.experiments.SampleParams
import utils.Params

import java.io.File
import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport

object SamplingExperiment {

  val samplingNames = Array("VocabSelect", "VotedDivergence", "KMeans", "KL", "VE", "LM", "Mahalonabis", "Euclidean", "Entropy", "KL", "Least", "Boltzmann")
  val adapterNames = Array("avg")

  val selectionSizes = Array(1000, 5000, 25000)
  val tasks = Array("ner", "pos", "sentiment", "intrinsic")
  val models = Array("cbow", "skip", "lstm")
  val jsonFilename = "resources/evaluations/sentence-tr.json"

  val embedParams = new SampleParams()

  def experimentKey(params: Params): Int = {
    val extractorName = "feature"
    val embeddingSize = params.embeddingLength
    val hiddenSize = params.hiddenLength
    val clusterSize = params.clusterSize
    val windowSize = params.windowLength
    val committee = params.committee
    val knn = params.knn
    val tokenLength = params.tokenLength
    val dictionarySize = params.dictionarySize


    val array = Array[Int](
      params.maxSelectSize,
      dictionarySize,
      embeddingSize,
      hiddenSize,
      windowSize,
      clusterSize,
      tokenLength,
      true.hashCode(),
      committee,
      knn,
      extractorName.hashCode,
      params.selectionMethod.hashCode)

    val keyID = array.foldRight[Int](7) { case (crr, main) => main * 7 + crr }
    keyID
  }

  def evaluate(embedParams: SampleParams, model: String, taskName: String): this.type = {
    if ("intrinsic".equals(taskName)) evaluateIntrinsic(embedParams, model)
    else if ("ner".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicNER(embedParams))
    else if ("pos".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicPOS(embedParams))
    else if ("sentiment".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicSentiment(embedParams))
    else null
  }

  def evaluateExtrinsic(embedParams: SampleParams, model: String, function: ExtrinsicLSTM): this.type = {


    val sentenceFilename = embedParams.sampledDataset()
    //embedParams.textFolder + function.getClassifier() +"/" + embedParams.selectionMethod + "-" + embedParams.maxSelectSize + "-" + key + ".txt"

    val modelling = model + "/" + embedParams.scorerName + "-" +embedParams.maxSelectSize + "/" +function.getClassifier() + "/" + embedParams.embeddingModel + "-" + embedParams.maxSelectSize + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Result filename: " + resultName + " Task: "+function.getClassifier())
      val mainEvaluation = new IntrinsicEvaluation(resultName)
      mainEvaluation.functions :+= function

      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel)
        .train(sentenceFilename)

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
    }
    else if (new File(resultName).exists()) {
      println("Found: " + resultName)
    }
    this
  }


  def evaluateIntrinsic(embedParams: SampleParams, model: String): this.type = {


    val sentenceFilename = embedParams.sampledDataset()
    //embedParams.intrinsicTextFolder + embedParams.scorerName + "-" + embedParams.maxSelectSize + "-" + key + ".txt"

    val modelling = model + "/" + embedParams.scorerName + "-" +embedParams.maxSelectSize + "/intrinsic/" + embedParams.embeddingModel + "-" + embedParams.maxSelectSize + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Result filename: " + resultName)
      val mainEvaluation = new IntrinsicEvaluation(resultName)
        .attachEvaluations(jsonFilename)
        .compile()


      mainEvaluation.filter(Array("SEMEVAL"))
      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel)
        .train(sentenceFilename)

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
    }
    else if (new File(resultName).exists()) {
      println("Found: " + resultName)
    }
    this
  }


  def evaluate(mainEvaluation: IntrinsicEvaluation, model: EmbeddingModel, params: SampleParams): Unit = {

    println("==============================================================")
    println("==========" + params.embeddingModel.toUpperCase + "===============")
    println("==============================================================")

    mainEvaluation.evaluateReport(model, params)
  }

  def main(args: Array[String]): Unit = {

   //System.setProperty("org.bytedeco.openblas.load", "mkl")

    val parCollection = samplingNames.par
    parCollection.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(1))
    parCollection.foreach(scorerName => {
      tasks.foreach(taskName => {
        selectionSizes.foreach(selectSize => {
          adapterNames.foreach(adapterName => {
            models.foreach(crrModel => {
              println(s"Evaluating ${scorerName} - ${selectSize} on task ${taskName}")
              val crrParams = embedParams.copy()
              crrParams.scorerName = scorerName
              crrParams.maxSelectSize = selectSize
              crrParams.adapterName = adapterName
              crrParams.embeddingModel = crrModel
              evaluate(crrParams, crrModel, taskName)
            })
          })
        })
      })
    })
  }
}
