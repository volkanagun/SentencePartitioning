package experiments

import transducer.LMExperiment

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

class LDPrunning {


  var pruneSizes = Array(/*Int.MaxValue, 1000,*/ 300 /*, 10*/)
  var topSample = Array(5 /*, 4, 3, 2, 1*/)
  var windows = Array(/*6, */ 4, 2 /*, 4, 2*/).reverse
  var adapters = Array(/*"frequent-ngram"*//*"lm-rank",*/ "lm-lemma", "lm-ngram", "lm-syllable", "lm-skip", "lm-rank-efficient")
  val tasks = Array("pos", "ner", "sentiment", "intrinsic")
  val doTask = true
  val doConstruct = false

  def evaluate(): this.type = {

    val dataset = new LMDataset()
    val experiment = new LMExperiment()
    val evaluator = new MainExperiment()

    pruneSizes.foreach(prune => {
      topSample.foreach(sample => {

        windows.foreach(window => {
          adapters.foreach(adapter => {
            val params = new Params()
            //global
            params.lmMaxSentence = 240
            params.lmMaxSentenceLength = 200
            params.lmMinSentenceLength = 50
            params.lmEpocs = 1000
            params.lmThreads = 128
            params.lmMaxWaitSeconds = 60
            params.lmSample = sample
            params.maxSentence = Int.MaxValue
            params.lmPrune = prune
            params.lmTopSplit = 5
            params.lmPrune = prune
            params.lmWindowLength = window
            params.adapterName = adapter
            params.epocs = 5
            params.batchSize = 96
            params.evalBatchSize = 48
            params.evalEpocs = 10
            params.lmDoSample = true
            params.lmDoPrune = true
            params.lmForceTrain = false
            params.lmTrainDictionary = true
            params.forceTrain = false
            params.forceEval = false
            params.corpusEpocs = 100
            params.corpusMaxSentence = 2400

            if (doConstruct) {
              experiment.train(params, adapter)
            }

            experiment.initGraphStats(params, adapter)

            tasks.foreach(task => {
              val copyParams = params.copy()
              if (doConstruct) {
                dataset.construct(copyParams, task)
              }
              System.gc()
              if (doTask) {
                evaluator.evaluate(copyParams, params.embeddingModel, task)
              }
            })

          })
        })
      })
    })

    //})

    this
  }


}

object LDPrunning {
  def main(args: Array[String]): Unit = {
    new LDPrunning().evaluate()
  }
}
