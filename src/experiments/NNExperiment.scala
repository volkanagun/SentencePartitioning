package experiments

import ch.qos.logback.classic.encoder.PatternLayoutEncoder
import ch.qos.logback.classic.{Level, LoggerContext}
import ch.qos.logback.core.ConsoleAppender
import evaluation._
import models.EmbeddingModel
import org.slf4j.{Logger, LoggerFactory}
import transducer.{AbstractLM, LMExperiment}
import utils.Tokenizer

import java.io.File
import scala.util.control.Breaks

class NNExperiment {


  val selectionSizes = Array(10000000)
  val tasks = Array("ner", "pos", "sentiment", "intrinsic").reverse

  val ranges = Array(6, 4, 2)
  val models = Array("cbow"/*, "skip"*//*, "self-attention"*/)
  val jsonFilename = "resources/evaluation/analogy/sentence-tr.json"
  val tokenizerFilename = "resources/dictionary/dictionary.zip"

  val tokenizer = new Tokenizer(windowSize = 2).loadZip(tokenizerFilename)
  val embedParams = new Params()

   def evaluate(embedParams: Params, model: String, taskName: String, lm:AbstractLM): Boolean = {
    if ("intrinsic".equals(taskName)) evaluateIntrinsic(embedParams, model, lm)
    else if ("ner".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicNER(embedParams, tokenizer, lm), lm)
    else if ("pos".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicPOS(embedParams, tokenizer, lm), lm)
    else if ("sentiment".equals(taskName)) evaluateExtrinsic(embedParams, model, new ExtrinsicSentiment(embedParams, tokenizer, lm), lm)
    else false
  }


  def evaluateExtrinsic(embedParams: Params, model: String, function: ExtrinsicLSTM,  lm:AbstractLM): Boolean = {

    val sentenceFilename = embedParams.corpusFilename(function.getClassifier())
    val modelling = model + "/" + embedParams.adapterName + "/" + function.getClassifier() + "/" + embedParams.embeddingModel + "/"
    val modelID = modelling.hashCode.toString
    val resultName = embedParams.resultFilename(modelling)

    if (!new File(resultName).exists() && new File(sentenceFilename).exists()) {
      println("Result filename: " + resultName + " Task: " + function.getClassifier())
      val mainEvaluation = new IntrinsicEvaluation(resultName)
      mainEvaluation.functions :+= function

      val words = mainEvaluation.universe()
      embedParams.modelName(model + "-" + modelID)

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel, tokenizer, lm)
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


  def evaluateIntrinsic(embedParams: Params, model: String,  lm:AbstractLM): Boolean = {

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

      val embeddingModel = embedParams.createModel(embedParams.embeddingModel, tokenizer, lm)
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
      val experiments = new NNExperiment()
      experiments.models.foreach(crrModel => {
        experiments.tasks.foreach(taskName => {
          params.adapters.foreach(adapterName => {
            experiments.selectionSizes.foreach(selectSize => {
              experiments.ranges.foreach(r => {
                println(s"Evaluating model ${crrModel} with ${selectSize} on task ${taskName}")
                val newParams = Params(adapterName, r)
                newParams.adapterName = adapterName
                newParams.embeddingModel = crrModel
                newParams.maxSentences = selectSize
                newParams.lmMaxSentence = selectSize
                newParams.epocs = 5
                newParams.batchSize = 128
                newParams.forceTrain = false
                experiments.evaluate(newParams, crrModel, taskName, null)
              })
            })
          })
        })
      })
    }
  }
}

