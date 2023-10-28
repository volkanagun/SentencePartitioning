package sampling.experiments

import java.io.File

class SampleParams {
  var dictionarySize = 1000000
  var secondDictionarySize = 10000
  var embeddingSize = 300
  var hiddenSize = 100
  var clusterSize = 20
  var ngramCombinationSize = 10
  var windowSize = 20
  var embeddingWindowSize = 5
  var committee = 10
  var batchSize = 24
  var nthreads = 1
  var topSelects = 1

  var knn = 7
  var tokenLength = 1
  var freqcutoff= 20

  var maxSentenceSize = 1000000000
  val max10million = 10000000
  val max100million = 100000000
  val max1billion = 1000000000
  val max1million = 1000000
  val max10thausand = 10000
  val max100thausand = 100000
  val max1thausand = 1000
  var maxSelectSize = 1000
  var threshold = 0.12
  var kmoving = 100
  var kselectSize = 1
  var maxInitSamples = 1000
  var maxWordSamples = 100

  var maxSentenceLength = 200
  var minSentenceLength = 50
  val textFolder = "resources/text/"
  val dictionaryFolder = "resources/dictionary/"
  var useWords = true
  val sentenceFilename = textFolder + "/sentences/sentences-tr.txt"

  var nerTask = "ner"
  var posTask = "pos"
  var sentimentTask = "sentiment"
  var intrinsicTask = "intrinsic"

  var nerDataset = textFolder + "ner/dataset.txt"
  var posDataset = textFolder + "pos/dataset.txt"
  var sentimentDataset = textFolder + "sentiment/dataset.txt"
  var intrinsicDataset = textFolder + "intrinsic/dataset.txt"

  var dataset = intrinsicDataset
  var taskName = intrinsicTask

  var extractorName: String = null
  var scorerName: String = null
  var adapterName:String=null


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
    copyParams

  }

  def evalDictionaryFile(task:String):String={
    if("sentiment".equals(task)) "resources/texts/dictionaries/sentiment.txt"
    else if("ner".equals(task)) "resources/texts/dictionaries/ner.txt"
    else if("pos".equals(task)) "resources/texts/dictionaries/pos.txt"
    else if("intrinsic".equals(task)) "resources/texts/dictionaries/intrinsic.txt"
    else null
  }

}
