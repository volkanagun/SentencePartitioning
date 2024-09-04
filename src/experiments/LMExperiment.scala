package transducer

import com.aparapi.device.{Device, OpenCLDevice}
import com.aparapi.internal.kernel.KernelManager
import com.aparapi.internal.opencl.OpenCLPlatform
import experiments.Params

import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

class LMExperiment {


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
  }

  def model(params: Params, name: String): AbstractLM = {

    val lm = if ("lm-skip".equals(name)) {
      new SkipLM(params)
    }
    else if ("lm-rank".equals(name)) {
      new RankLM(params)
    }
    else if ("lm-rank-efficient".equals(name)) {
      new LMRankEfficient(params)
    }
    else if ("lm-rank-gpu".equals(name)) {
      new RankGPULM(params)
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
    else if ("lm-word".equals(name)) {
      new WordLM(params)
    }
    else if ("lm-char".equals(name)) {
      new CharLM(params)
    }
    else {
      null
    }


    if(lm.exists()) lm.load()
    else null
  }

  def model(name: String, window: Int): AbstractLM = {

    val params = init(name, window)

    val lm = if ("lm-skip".equals(name)) {
      new SkipLM(params)
    }
    else if ("lm-rank".equals(name)) {
      new RankLM(params)
    }
    else if ("lm-rank-efficient".equals(name)) {
      new LMRankEfficient(params)
    }
    else if ("lm-rank-gpu".equals(name)) {
      new RankGPULM(params)
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
    else if ("lm-word".equals(name)) {
      new WordLM(params)
    }
    else if ("lm-char".equals(name)) {
      new CharLM(params)
    }
    else {
      null
    }


    lm.load()
    //lm.test()
    lm
  }

  def train(): Unit = {

    val params = new Params()
    val ranges = params.windows
    val models = params.adapters
    ranges.foreach(window => {
      models.foreach(name => {
        val lm = model(name, window)
        if (!lm.exists() || params.lmForceTrain) {
          println("Training LM model: " + name + " with window: " + window)
          lm.initialize()
            .loadTrain()
        }
      })
    })

  }

  def train(params: Params, name: String): Unit = {

    val lm = model(params, name)
    if (!lm.exists() || params.lmForceTrain) {
      println("Training LM model: " + name + " with window: " + params.lmWindowLength)
      lm.initialize()
        .loadTrain()
    }
    else {
      println("Model found: " + lm.exists())
    }

  }

  def initGraphStats(params: Params, name:String) : Params = {
    val lm = model(params, name)

    val mapping = if(lm.exists()) lm.load().graphStats()
    else Map[String, Double]()

    params.setGraphStats(mapping)
  }

  def experimentSkipLM(): Unit = {
    model("skipLM", 4)
      .initialize()
      .loadTrain()
  }

  def experimentRankLM(): Unit = {
    model("rankLM", 4)
      .initialize()
      .loadTrain()
  }
}

object LMExperiment extends LMExperiment {

  System.setProperty("com.aparapi.executionMode", "GPU")
  System.setProperty("com.aparapi.enableExecutionModeReporting", "true")
  System.setProperty("com.aparapi.enableProfiling", "true")
  System.setProperty("java.library.path", "/usr/lib/x86_64-linux-gnu/")
  System.setProperty("com.amd.aparapi.useByteBuffers", "false")


  def list(): Array[Device] = {
    val preferences = KernelManager.instance.getDefaultPreferences
    preferences.getPreferredDevices(null).toArray[Device](Array[Device]())
  }
  def main(_args: Array[String]): Unit = {


    System.out.println("com.aparapi.examples.info.Main")

    val platforms = list()
    System.out.println("Machine contains " + platforms.size + " OpenCL platforms")
    var platformc = 0

    for (platform <- platforms) {
      System.out.println("Platform " + platformc + "{")
      System.out.println("   Name    : \"" + platform.getType.name() + "\"")

      System.out.println("}")
      platformc += 1
    }
    val preferences = KernelManager.instance.getDefaultPreferences()
    System.out.println("\nDevices in preferred order:\n")
    val deviceArray = preferences.getPreferredDevices(null).toArray[Device](Array[Device]())

    for (device <- deviceArray) {

      System.out.println(device)
      System.out.println()
    }
  }
}
