package experiments

import models.{CBOWModel, EmbeddingModel, FastTextModel, GloveModel, SelfAttentionLSTM, SkipGramModel}
import transducer.{AbstractLM, FrequentLM, LMSubWord, LemmaLM, NGramLM, RankLM, SkipLM, SyllableLM, WordLM}
import utils.Tokenizer

class Params {

  val adapters = Array("lm-subword", "lm-rank", "lm-lemma", /*"frequent-ngram",*/ "lm-ngram", "lm-syllable", "lm-skip").reverse
  val windows = Array(2, 3, 4).reverse
  val tasks = Array("ner", "pos", "sentiment", "intrinsic")
  val selectionSizes = Array(500000)

  var graphStats = Map[String, Double]()

  var epocs: Int = 5000

  var batchSize: Int = 1

  var lrate = 0.001
  var hiddenLength: Int = 200
  var nheads: Int = 4
  var embeddingLength: Int = 100
  var embeddingWindowLength: Int = 20
  var forceTrain = false
  var forceEval = true

  var dictionarySize: Int = 500000
  var freqCutoff: Int = 0


  var evalWindowLength: Int = 200
  var evalUseEmbeddings: Boolean = true
  var evalDictionarySize = 500000
  var evalBatchSize = 24
  var evalEpocs = 15

  var sentimentSize: Int = 3
  var nerSize: Int = 10
  var posSize: Int = 12


  var maxSentences = 10000000
  var lmMaxSentence = 200000


  var adapterName = "skip"
  var tokenLength = 5
  var windowLength = 10

  var minSentenceLength = 0
  var maxSentenceLength = 500
  var lmMaxSentenceLength = 300
  var lmMinSentenceLength = 70
  var lmTokenLength = 25

  var maxSentence = 4096
  var lmTrainDictionary = true
  var lmepocs = 1
  var lmrange = 10000
  var lmEpocs: Int = 5
  var corpusEpocs: Int = 100
  var corpusMaxSentence: Int = 2400

  var lmPrune: Int = 100
  var lmWindowLength = 10
  var lmMaxWaitSeconds = 20

  var lmSlideLength = 7
  var lmCandidateCount = 3
  var lmStemLength = 7
  var lmSkip = 20
  var lmSample = 10
  var lmTopSplit = 10
  var lmTopSyllableSplit = 200
  var lmIterations = 10
  var lmForceTrain = false
  var lmDoPrune = true
  var lmDoSample = false
  var name: String = "modelName"

  var embeddingModel = "cbow"
  var nthreads = 4
  var lmThreads = 256

  val modelFolder = "resources/models/"
  val textFolder = "resources/text/"
  var sentencesFile = "resources/text/sentences/sentences-tr.txt"
  val embeddingFolder = "resources/embeddings/"
  val intrinsicTextFolder = "resources/text/intrinsic/"

  val evaluationFolder = "resources/evaluation/"
  val dictionaryFolder = "resources/dictionary/"
  val resultFolder = "resources/results/"
  val fastTextBin = "resources/binary/cc.tr.300.vec"

  var stats: LMStats = new LMStats()

  def setStats(stats: LMStats): this.type = {
    this.stats.totalTokenCount = stats.totalTokenCount
    this.stats.totalSentenceCount = stats.totalSentenceCount
    this.stats.totalTime = stats.totalTime
    this
  }

  def setGraphStats(graphStats: Map[String, Double]): this.type = {
    this.stats.setGraphStatistics(graphStats)
    this
  }

  def copy(): Params = {
    val params = new Params()
    params.embeddingModel = embeddingModel
    params.adapterName = adapterName

    params.epocs = epocs
    params.batchSize = batchSize
    params.lrate = lrate
    params.hiddenLength = hiddenLength
    params.nheads = nheads
    params.embeddingLength = embeddingLength
    params.embeddingWindowLength = embeddingWindowLength
    params.dictionarySize = dictionarySize
    params.freqCutoff = freqCutoff
    params.evalWindowLength = evalWindowLength
    params.sentimentSize = sentimentSize
    params.maxSentences = maxSentences
    params.tokenLength = tokenLength
    params.windowLength = windowLength
    params.minSentenceLength = minSentenceLength
    params.maxSentenceLength = maxSentenceLength
    params.nthreads = nthreads

    params.evalEpocs = evalEpocs
    params.evalDictionarySize = evalDictionarySize
    params.evalWindowLength = evalWindowLength
    params.evalBatchSize = evalBatchSize
    params.evalUseEmbeddings = evalUseEmbeddings

    params.lmEpocs = lmEpocs
    params.lmThreads = lmThreads
    params.lmMaxSentence = lmMaxSentence
    params.lmMaxSentenceLength = lmMaxSentenceLength
    params.lmTrainDictionary = lmTrainDictionary
    params.lmForceTrain = lmForceTrain
    params.lmDoSample = lmDoSample
    params.lmTopSplit = lmTopSplit
    params.lmTopSyllableSplit = lmTopSyllableSplit
    params.lmSample = lmSample
    params.lmSkip = lmSkip
    params.lmStemLength = lmStemLength
    params.lmWindowLength = lmWindowLength
    params.lmDoPrune = lmDoPrune
    params.lmPrune = lmPrune
    params.lmMinSentenceLength = lmMinSentenceLength
    params.lmTokenLength = lmTokenLength
    params.lmSlideLength = lmSlideLength
    params.lmCandidateCount = lmCandidateCount
    params.lmIterations = lmIterations
    params.lmMaxWaitSeconds = lmMaxWaitSeconds
    params.lmThreads = lmThreads
    params.stats = stats

    params
  }

  def modelDefault(name:String, window:Int):AbstractLM={
    val params = Params(name, window)
    model(params, name)
  }


  def model(params: Params, name: String): AbstractLM = {

    val lm = if ("lm-skip".equals(name)) {
      new SkipLM(params)
    }
    else if ("lm-rank".equals(name)) {
      new RankLM(params)
    }
    else if ("lm-subword".equals(name)) {
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
    else if ("lm-word".equals(name)) {
      new WordLM(params)
    }
    else {
      null
    }
    if(lm.exists()) lm.load()
    else lm
  }


  def createModel(name: String, tokenizer: Tokenizer,  lm:AbstractLM): EmbeddingModel = {
    val embeddingModel = if (name.contains("cbow")) new CBOWModel(this, tokenizer, lm)
    else if (name.contains("skip")) new SkipGramModel(this, tokenizer, lm)
    else if (name.contains("self")) new SelfAttentionLSTM(this, tokenizer, lm)
    else if (name.contains("fast")) new FastTextModel(this, tokenizer, lm)
    else if (name.contains("glove")) new GloveModel(this, tokenizer, lm)
    else null

    embeddingModel
  }

  def modelName(name: String): this.type = {
    embeddingModel = name
    this
  }

  def densityFilename(classifier:String):String={
    textFolder + "/" + classifier + "/" + lmID() + ".density"
  }

  def embeddingsFilename(): String = {
    embeddingFolder + modelID() + ".bin"
  }

  def corpusFilename(task: String): String = {
    textFolder + "/" + task + "/"+ lmID() + ".txt"
  }

  def mainCorpusFilename(classifier: String): String = {
    textFolder + "/" + classifier + "/main.txt"
  }

  def resultFilename(modelling: String): String = {
    val fname = resultFolder + modelling + "/" + lmMaxSentence + "-" + lmWindowLength
    fname + "-" + modelID() + ".xml"
  }

  def modelFilename(): String = {
    modelFolder + modelID() + ".zip"
  }

  def modelEvaluationFilename(): String = {
    modelFolder + "eval-" + modelID() + ".zip"
  }

  def dictionaryFilename(): String = {
    modelFolder + modelID() + ".bin"
  }
  def dictionaryZipFilename(): String = {
    modelFolder + modelID() + ".zip"
  }

  def modelID(): Int = {
    Array[Int](corpusID(), lrate.hashCode(), embeddingModel.hashCode, hiddenLength, nheads, embeddingLength, embeddingWindowLength, dictionarySize, lmID())
      .foldRight(7) { case (a, main) => a + 7 * main }
  }


  def corpusID(): Int = {
    Array[Int](lmID(), adapterName.hashCode, windowLength, tokenLength, maxSentences, minSentenceLength).foldRight(7) { case (a, main) => a + 7 * main }
  }

  def lmID(): Int = {
    var array = Array[Int](adapterName.hashCode, lmWindowLength, lmSlideLength, lmTopSplit, lmSkip, lmStemLength)
    if (lmDoPrune) array = array :+ lmPrune
    if (lmDoSample) array = array :+ lmSample

    array.foldRight(7) { case (a, main) => a + 7 * main }
  }


  def tag(name: String, value: String): String = {
    s"<PARAM LABEL=\"${name}\" VALUE=\"${value}\"/>\n"
  }

  def statTag(): String = {
    "<EFFICIENCY>\n" +
      tag("TOKEN_AVG", stats.avgTokenTime().toString) +
      tag("SENTENCE_AVG", stats.avgSentenceTime().toString) +
      "</EFFICIENCY>\n" +
      "<STATISTICS>\n" +
      stats.statMap.map { case (tagName, tagValue) => {
        tag(tagName, tagValue.toString)
      }
      }.mkString +
      "</STATISTICS>\n"
  }

  def modelTag(): String = {
    "<MODEL_PARAMETERS>\n" +
      tag("MODEL", embeddingModel) +
      tag("EPOCS", epocs.toString) +
      tag("LEARNING_RATE", lrate.toString) +
      tag("HIDDEN_LENGTH", hiddenLength.toString) +
      tag("EMBEDDING_LENGTH", embeddingLength.toString) +
      tag("WINDOW_LENGTH", embeddingWindowLength.toString) +
      tag("HEADS", nheads.toString) +
      tag("DICTIONARY_SIZE", dictionarySize.toString) +
      "</MODEL_PARAMETERS>\n"
  }

  def partitionTag(): String = {
    "<PARTITION_PARAMETERS>\n" +
      tag("PARTITION_NAME", adapterName) +
      tag("MAX_SENTENCES", maxSentences.toString) +
      tag("TOKEN_LENGTH", tokenLength.toString) +
      tag("WINDOW_LENGTH", windowLength.toString) +
      tag("MIN_SENTENCE_LENGTH", minSentenceLength.toString) +
      tag("MAX_SENTENCE_LENGTH", maxSentenceLength.toString) +
      tag("NTHREADS", nthreads.toString) +
      "</PARTITION_PARAMETERS>\n"
  }

  def languageTag(): String = {
    "<LANGUAGE_PARAMETERS>\n" +
      tag("PARTITION_NAME", adapterName) +
      tag("LM_EPOCS", lmEpocs.toString) +
      tag("LM_ITERATIONS", lmIterations.toString) +
      tag("LM_MAXSENTENCES", lmMaxSentence.toString) +
      tag("LM_THREADS", lmThreads.toString) +
      tag("LM_TOP_SPLIT", lmTopSplit.toString) +
      tag("LM_PRUNE", lmPrune.toString) +
      tag("LM_DOSAMPLE", lmDoSample.toString) +
      tag("LM_SAMPLE", lmSample.toString) +
      tag("LM_SLIDE_LENGTH", lmSlideLength.toString) +
      tag("LM_WINDOW_LENGTH", lmWindowLength.toString) +
      tag("LM_FORCE_TRAIN", lmForceTrain.toString) +
      tag("LM_SKIP", lmSkip.toString) +
      tag("LM_STEM_LENGTH", lmStemLength.toString) +
      tag("LM_MAX_SENTENCE_LENGTH", lmMaxSentenceLength.toString) +
      "</LANGUAGE_PARAMETERS>\n"
  }

  def toShortXML(): String = {
    modelTag() +
      statTag() +
      partitionTag() +
      languageTag()
  }
}

object Params{

  def apply(name: String, window: Int): Params = {
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

 def apply(name: String): Params = {
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
    params.lmTopSplit = 3
    params.lmForceTrain = false



    params
  }


}