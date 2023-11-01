package sampling.experiments

import models.{CBOWModel, EmbeddingModel, SelfAttentionLSTM, SkipGramModel}

import java.io.File

class SampleParams {
  var dictionarySize = 100000
  var secondDictionarySize = 5000
  var embeddingSize = 300
  var hiddenSize = 20
  var clusterSize = 20
  var ngramCombinationSize = 10
  var windowSize = 20
  var embeddingWindowSize = 10
  var committee = 10
  var batchSize = 128
  var nthreads = 1
  var topSelects = 1

  var knn = 7
  var tokenLength = 5
  var freqcutoff= 10

  var maxSentenceSize = max100million
  val max10million = 10000000
  val max100million = 100000000
  val max1billion = 1000000000
  val max1million = 1000000
  val max10thausand = 10000
  val max100thausand = 100000
  val max1thausand = 1000
  var maxSelectSize = max100thausand
  var threshold = 0.015
  var kmoving = 200
  var kselectSize = 10
  var maxInitSamples = 1000
  var maxWordSamples = 10000

  var maxSentenceLength = 200
  var minSentenceLength = 50
  val textFolder = "resources/text/"
  val dictionaryFolder = "resources/dictionary/"
  val binaryFolder = "resources/binary/"
  var useWords = true
  val sentenceFilename = textFolder + "/sentences/sentences-tr.txt"

  var nerTask = "ner"
  var posTask = "pos"
  var sentimentTask = "sentiment"
  var intrinsicTask = "intrinsic"

  var intrinsicDataset = textFolder + "intrinsic/dataset.txt"

  val intrinsicTextFolder = "resources/text/intrinsic/"


  val evaluationFolder = "resources/evaluation/"
  val resultFolder = "resources/results/"
  val modelFolder = "resources/models/"

  var dataset = intrinsicDataset
  var taskName = intrinsicTask

  var extractorName: String = "feature"
  var scorerName: String = null
  var adapterName:String="avg"


  //Embedding extraction parameters
  var embeddingModel = "cbow"
  var embeddingLength = 100
  var hiddenLength = 100
  var modelWindowLength:Int = 20
  var nheads:Int = 4
  var freqCutoff:Int = 0
  var epocs = 10
  var lrate = 0.005



  //Evaluation parameters
  var evalWindowLength = 20

  def modelName(name: String): this.type = {
    embeddingModel = name
    this
  }

  def createModel(name: String): EmbeddingModel = {
    if (name.startsWith("cbow")) new CBOWModel(this)
    else if (name.startsWith("skip")) new SkipGramModel(this)
    else if (name.startsWith("lstm")) new SelfAttentionLSTM(this)
    else null
  }

  def sampledDataset():String={
    val folder = textFolder + taskName +"/"
    val keyFilename = folder + scorerName + "-"+ maxSelectSize + "-" + experimentKey() + ".txt"
    new File(folder).mkdirs()
    keyFilename
  }

  def mainDataset():String={
    val keyFilename = textFolder + taskName +"/" + "main.txt"
    keyFilename
  }

  def dictionaryFile():String={
    dictionaryFolder + taskName + ".txt"
  }

  def resultFilename(modelling: String): String = {
    val fname = resultFolder + modelling
    fname + "/" + modelID() + ".xml"
  }

  def experimentKey(): Int = {
    val array = Array[Int](
      maxSelectSize,
      dictionarySize,
      secondDictionarySize,
      embeddingSize,
      hiddenSize,
      windowSize,
      ngramCombinationSize,
      clusterSize,
      tokenLength,
      useWords.hashCode(),
      committee,
      knn,
      taskName.hashCode,
      extractorName.hashCode,
      scorerName.hashCode,
      adapterName.hashCode)

    val keyID = array.foldRight[Int](7) { case (crr, main) => main * 7 + crr }

    keyID
  }

  def copy():SampleParams={
    val copyParams = new SampleParams()
    copyParams.kmoving = kmoving
    copyParams.kselectSize = kselectSize
    copyParams.maxSelectSize = maxSelectSize
    copyParams.embeddingSize = embeddingSize
    copyParams.hiddenSize = hiddenSize
    copyParams.dictionarySize = dictionarySize
    copyParams.secondDictionarySize =secondDictionarySize
    copyParams.windowSize = windowSize
    copyParams.maxSentenceSize = maxSentenceSize

    copyParams.minSentenceLength = minSentenceLength
    copyParams.maxSentenceLength = maxSentenceLength
    copyParams.tokenLength = tokenLength

    copyParams.batchSize = batchSize
    copyParams.clusterSize = clusterSize
    copyParams.committee = committee
    copyParams.knn = knn

    copyParams.adapterName = adapterName
    copyParams.extractorName = extractorName
    copyParams.scorerName = scorerName
    copyParams.threshold = threshold
    copyParams.topSelects = topSelects
    copyParams.useWords = useWords
    copyParams.freqcutoff = freqcutoff


    copyParams.embeddingModel = embeddingModel
    copyParams.embeddingLength = embeddingLength
    copyParams.hiddenLength = hiddenLength
    copyParams.modelWindowLength = modelWindowLength
    copyParams.nheads = nheads
    copyParams.freqcutoff = freqcutoff
    copyParams.epocs = epocs
    copyParams.lrate = lrate
    copyParams.nthreads = nthreads


    copyParams

  }

  def evalDictionaryFile(task:String):String={
    if("sentiment".equals(task)) "resources/texts/dictionaries/sentiment.txt"
    else if("ner".equals(task)) "resources/texts/dictionaries/ner.txt"
    else if("pos".equals(task)) "resources/texts/dictionaries/pos.txt"
    else if("intrinsic".equals(task)) "resources/texts/dictionaries/intrinsic.txt"
    else null
  }

  def modelFilename(): String = {
    modelFolder + modelID() + ".bin"
  }

  def dictionaryFilename(): String = {
    binaryFolder + modelID() + ".bin"
  }

  def modelID(): Int = {
    Array[Int](lrate.hashCode(), embeddingModel.hashCode, hiddenLength, nheads, embeddingLength, modelWindowLength, dictionarySize).foldRight(7) { case (a, main) => a + 7 * main }
  }

  def corpusID(): Int = {
    Array[Int](maxSelectSize, scorerName.hashCode, adapterName.hashCode).foldRight(7) { case (a, main) => a + 7 * main }
  }

  def tag(name: String, value: String): String = {
    s"<PARAM LABEL=\"${name}\" VALUE=\"${value}\"/>\n"
  }

  def toShortXML(): String = {
    "<MODEL_PARAMETERS>\n" +
      tag("MODEL", embeddingModel) +
      tag("EPOCS", epocs.toString) +
      tag("LEARNING_RATE", lrate.toString) +
      tag("HIDDEN_LENGTH", hiddenLength.toString) +
      tag("EMBEDDING_LENGTH", embeddingLength.toString) +
      tag("WINDOW_LENGTH", modelWindowLength.toString) +
      tag("HEADS", nheads.toString) +
      tag("DICTIONARY_SIZE", dictionarySize.toString) +
      tag("SELECTION_SIZE", maxSelectSize.toString) +
      "</MODEL_PARAMETERS>\n" +
      "<SELECTION_PARAMETERS>\n" +
      tag("SELECTION_METHOD", scorerName) +
      tag("ADAPTER_NAME", adapterName) +
      //tag("MAX_SENTENCES", maxSentences.toString) +
      tag("MAX_SELECT_SIZE", maxSelectSize.toString) +
      //tag("SELECTION_SIZE", selectionSize.toString) +
      tag("COMITTEE", committee.toString) +
      tag("TOKEN_LENGTH", tokenLength.toString) +
      tag("CLUSTER_SIZE", clusterSize.toString) +
      tag("KNN", knn.toString) +
      tag("WINDOW_LENGTH", windowSize.toString) +
      tag("MIN_SENTENCE_LENGTH", minSentenceLength.toString) +
      tag("MAX_SENTENCE_LENGTH", maxSentenceLength.toString) +
      tag("NTHREADS", nthreads.toString) +
      "</SELECTION_PARAMETERS>\n"
  }

}
