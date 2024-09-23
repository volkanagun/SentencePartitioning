package transducer

import com.aparapi.device.{Device, OpenCLDevice}
import com.aparapi.internal.kernel.KernelManager
import com.aparapi.internal.opencl.OpenCLPlatform
import experiments.Params

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

class LMExperiment {
/*

  def init(name: String, window: Int): Params = {
    val params = new Params()
    params.adapterName = name

    params.lmMaxSentence = 240
    params.lmMaxSentenceLength = 200
    params.lmTokenLength = 25
    params.lmrange = 2000
    params.lmEpocs = 1000
    params.lmTrainDictionary = true
    params.lmPrune = 100
    params.lmThreads = 48

    params.corpusEpocs = 100
    params.corpusMaxSentence = 2400

    params.lmWindowLength = window
    params.lmTopSplit = 3
    params.lmSkip = window
    params.lmSlideLength = window
    params.lmForceTrain = false



    params
  }*/



 /* def model(name: String, window: Int): AbstractLM = {

    val params = init(name, window)

    val lm = if ("lm-skip".equals(name)) {
      new SkipLM(params)
    }
    else if ("lm-rank".equals(name)) {
      new RankLM(params)
    }
    else if ("lm-rank-efficient".equals(name)) {
      new LMSubWord(params)
    }
    else if ("lm-ngram".equals(name)) {
      new NGramLM(params)
    }
    else if ("lm-syllable".equals(name)) {
      new SyllableLM(params)
    }
    else if ("lm-lemma".equals(name)) {
      new LemmaLM(params)
    }
    else if ("frequent-ngram".equals(name)) {
      new FrequentLM(params)
    }
    else {
      null
    }


    lm.load()
    //lm.test()
    lm
  }
*//*
  def train(): Unit = {

    val params = new Params()
    val ranges = params.windows
    val models = params.adapters
    ranges.foreach(window => {
      models.foreach(name => {
        val lm = model(name, window)
        if (!lm.exists() || params.lmForceTrain) {
          println("Training LM model: " + name + " with window: " + window)
          lm.initialize().loadTrain()
        }
      })
    })

  }*/
/*

  def train(params: Params, name: String): AbstractLM = {

    val lm = params.model(params, name)
    if (!lm.exists() || params.lmForceTrain) {
      println("Training LM model: " + name + " with window: " + params.lmWindowLength)
      lm.initialize()
      lm.loadTrain()
    }
    else {
      println("Model found: " + lm.exists())
    }

    lm
  }

  def initGraphStats(params: Params, name:String) : Params = {
    val lm = params.model(params, name)

    val mapping = if(lm.exists()) lm.load().graphStats()
    else Map[String, Double]()

    params.setGraphStats(mapping)
  }
*/


}
