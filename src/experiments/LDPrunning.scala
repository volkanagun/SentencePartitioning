package experiments

import org.nd4j.linalg.factory.Nd4j
import transducer.{AbstractLM, LMExperiment}

import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport

class LDPrunning {


  var pruneSizes = Array(1000/*, 300*/)
  var topSample = Array(5)
  var windows = Array(3)
  var adapters = Array(/*"frequent-ngram", "lm-word", */"lm-subword", "lm-syllable", "lm-lemma", "lm-skip", "lm-rank").reverse
  val tasks = Array(/*"pos","ner", "sentiment",*/"intrinsic")
  val embeddingModels = Array("fast"/*,"cbow", "skip","glove"*/)
  val embeddingLengths = Array(100, 100, 100)
  val doTask = true
  val doConstruct = true


  def parameters(): Array[Params] = {
    embeddingModels.zip(embeddingLengths).flatMap(embeddingModelPair => {
      val embeddingModel = embeddingModelPair._1
      val embeddingLength = embeddingModelPair._2

      pruneSizes.flatMap(prune => {
        topSample.flatMap(sample => {

          windows.flatMap(window => {
            adapters.map(adapter => {
              val params = new Params()
              //global
              params.embeddingModel = embeddingModel
              params.embeddingLength = embeddingLength
              params.lmMaxSentence = 240

              params.dictionarySize = 100000
              params.lmMaxSentenceLength = 200
              params.lmMinSentenceLength = 50
              params.lmIterations = 1
              params.lmEpocs = 1000
              params.lmThreads = 240
              params.lmMaxWaitSeconds = 60
              params.lmSample = sample
              params.maxSentence = Int.MaxValue
              params.lmPrune = prune
              params.lmTopSplit = 5
              params.lmPrune = prune
              params.lmWindowLength = window
              params.adapterName = adapter
              params.epocs = 5
              params.batchSize = 48
              params.evalBatchSize = 96
              params.evalEpocs = 10
              params.lmDoSample = true
              params.lmDoPrune = true
              params.lmForceTrain = false
              params.lmTrainDictionary = false
              params.forceTrain = false
              params.forceEval = false
              params.corpusEpocs = 100
              params.corpusMaxSentence = 2400
              params
            })
          })
        })
      })
    })

  }

  def createDensity(): this.type = {
    val ps = parameters()
    val dataset = new LMDataset()
    ps.foreach(params => {
      tasks.par.foreach(taskName => {
        dataset.densityScores(params, taskName)
      })
    })
    this
  }

  def train(): Unit = {

    val params = new Params()
    val ranges = params.windows
    val models = params.adapters
    ranges.foreach(window => {
      models.foreach(name => {
        val lm = params.modelDefault(name, window)
        if (!lm.exists() || params.lmForceTrain) {
          println("Training LM model: " + name + " with window: " + window)
          lm.initialize().loadTrain()
        }
      })
    })
  }

  def train(params: Params, name: String): AbstractLM = {

    val lm = params.model(params, name)
    if (!lm.exists() || lm.isEmpty() || params.lmForceTrain) {
      println("Training LM model: " + name + " with window: " + params.lmWindowLength)
      lm.initialize()
      lm.loadTrain()
    }
    else {
      println("Model found: " + lm.exists())
    }

    lm
  }



  def evaluate(): this.type = {

    val dataset = new LMDataset()
    val evaluator = new NNExperiment()

    val paramss = parameters()
    paramss.foreach(params => {


      val lm = train(params, params.adapterName)
      params.setGraphStats(lm.graphStats())

      tasks.foreach(task => {
        val copyParams = params.copy()
        if (doConstruct) {
          dataset.construct(lm, task)
        }

        System.gc()
        if (doTask) {
          evaluator.evaluate(copyParams, params.embeddingModel, task, lm)
        }
      })
    })

    this
  }


}

object LDPrunning {
  def main(args: Array[String]): Unit = {
    System.setProperty("org.bytedeco.openblas.load", "mkl")
    Nd4j.getEnvironment.setMaxPrimaryMemory(160L * 1024L * 1024L * 1024L)
    Nd4j.getEnvironment.setMaxDeviceMemory(340L * 1024L * 1024L * 1024L)
    new LDPrunning().evaluate()
  }
}
