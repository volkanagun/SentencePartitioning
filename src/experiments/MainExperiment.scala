package experiments

import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.{Level, LoggerContext}
import ch.qos.logback.core.ConsoleAppender
import evaluation._
import models.EmbeddingModel
import org.slf4j.{Logger, LoggerFactory}
import transducer.LMExperiment
import utils.Tokenizer

import java.io.File
import scala.util.control.Breaks

class MainExperiment {


  val selectionSizes = Array(10000000)
  val tasks = Array("ner", "pos", "sentiment", "intrinsic").reverse

  val ranges = Array(6, 4, 2)
  val models = Array("cbow"/*, "skip"*//*, "self-attention"*/)
  val jsonFilename = "resources/evaluation/analogy/sentence-tr.json"

  val tokenizer = new Tokenizer(windowSize = 2).loadZip()
  val embedParams = new Params()

  def experimentKey(params: Params): Int = {
    val extractorName = "feature"
    val embeddingSize = params.embeddingLength
    val hiddenSize = params.hiddenLength
    val windowSize = params.windowLength
    val tokenLength = params.tokenLength
    val dictionarySize = params.dictionarySize


    val array = Array[Int](
      dictionarySize,
      embeddingSize,
      hiddenSize,
      windowSize,
      tokenLength,
      true.hashCode(),
      extractorName.hashCode)

    val keyID = array.foldRight[Int](7) { case (crr, main) => main * 7 + crr }
    keyID
  }

  def evaluate(embedParams: Params, model: String, taskName: String): Boolean = {
    if ("intrinsic".equals(taskName)) evaluateIntrinsic(embedParams, model)
    else if ("ner".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicNER(embedParams, tokenizer))
    else if ("pos".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicPOS(embedParams, tokenizer))
    else if ("sentiment".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicSentiment(embedParams, tokenizer))
    else false
  }

  def constructCorpus(embedParams: Params, classifier: String): Unit = {
    new LMDataset().construct(embedParams, classifier)
  }

  def evaluateExtrinsic(embedParams: Params, model: String, function: ExtrinsicLSTM): Boolean = {

    val sentenceFilename = embedParams.corpusFilename(function.getClassifier())
    //embedParams.textFolder + function.getClassifier() +"/" + embedParams.selectionMethod + "-" + embedParams.maxSelectSize + "-" + key + ".txt"

    val modelling = model + "/" + embedParams.adapterName + "/" + function.getClassifier() + "/" + embedParams.embeddingModel + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Result filename: " + resultName + " Task: " + function.getClassifier())
      val mainEvaluation = new IntrinsicEvaluation(resultName)
      mainEvaluation.functions :+= function

      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel, tokenizer)
        .train(sentenceFilename)

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
      true
    }
    else if (new File(resultName).exists()) {
      println("Found: " + resultName)
      true
    }
    else {
      false
    }
  }


  def evaluateIntrinsic(embedParams: Params, model: String): Boolean = {

    val sentenceFilename = embedParams.corpusFilename("intrinsic")

    val modelling = model + "/" + embedParams.adapterName + "/intrinsic/" + embedParams.embeddingModel + "/"
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

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel, tokenizer)
        .train(sentenceFilename)

      mainEvaluation.setDictionary(words, embeddingModel)
      evaluate(mainEvaluation, embeddingModel, embedParams)
      true
    }
    else if (new File(resultName).exists()) {
      println("Found: " + resultName)
      true
    }
    else {
      false
    }
  }


  def evaluate(mainEvaluation: IntrinsicEvaluation, model: EmbeddingModel, params: Params): Unit = {

    println("==============================================================")
    println("==========" + params.embeddingModel.toUpperCase + "===============")
    println("==============================================================")
    mainEvaluation.evaluateReport(model, params)
  }

  def evaluate(): Unit = {
    val breaks = new Breaks()
    val params = new Params()
    breaks.breakable {
      val experiments = new MainExperiment()
      experiments.models.foreach(crrModel => {
        experiments.tasks.foreach(taskName => {
          params.adapters.foreach(adapterName => {
            experiments.selectionSizes.foreach(selectSize => {
              experiments.ranges.foreach(r => {
                println(s"Evaluating model ${crrModel} with ${selectSize} on task ${taskName}")
                val newParams = new LMExperiment().init(adapterName, r)
                newParams.adapterName = adapterName
                newParams.embeddingModel = crrModel
                newParams.maxSentences = selectSize
                newParams.lmMaxSentence = selectSize
                newParams.epocs = 5
                newParams.batchSize = 128
                newParams.forceTrain = false
                experiments.evaluate(newParams, crrModel, taskName)
              })
            })
          })
        })
      })
    }
  }
}

object MainExperiment {

  def main(args: Array[String]): Unit = {
    System.setProperty("org.bytedeco.openblas.load", "mkl")
    new MainExperiment().evaluate()
  }




}
