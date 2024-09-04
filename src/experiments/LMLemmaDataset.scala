package experiments

import tagging.hmm.SentenceHMM
import transducer.{AbstractLM, LemmaLM}
import utils.Tokenizer

import java.io.{File, FileOutputStream, PrintWriter}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source
import scala.util.Random

class LMLemmaDataset {

  val tokenizer = new Tokenizer()
  val sentenceHMM = new SentenceHMM().load()

  def init(name: String, window: Int): Params = {
    val params = new Params()
    params.adapterName = name
    params.lmTrainDictionary = true
    params.lmMaxSentence = 10000
    params.lmMaxSentenceLength = 250
    params.lmMinSentenceLength = 70
    params.lmrange = 5000
    params.lmEpocs = 2400
    params.lmWindowLength = window
    params.lmTopSplit = 5
    params.lmSkip = window - 1
    params.lmSlideLength = 5
    params.lmForceTrain = true
    params.lmThreads = 12

    params.maxSentences = 10000
    params.maxSentenceLength = 220
    params.epocs = 50000

    params
  }

  def initTrain(name: String): Params = {
    val params = new Params()
    params.adapterName = name
    params.lmTrainDictionary = true
    params.lmMaxSentence = 10000
    params.lmMaxSentenceLength = 120
    params.lmMinSentenceLength = 50
    params.lmrange = 5000
    params.lmEpocs = 2400
    params.lmThreads = 12
    params.lmWindowLength = 20
    params.lmSlideLength = 7
    params.lmTopSplit = 3
    params.lmSkip = 10
    params.lmForceTrain = true

    params.maxSentences = 20000
    params.maxSentenceLength = 200
    params.epocs = 50000


    params
  }

  def train(textFile: String): LemmaLM = {
    val lemmaLM = new LemmaLM(initTrain("lemmaBigLM"))
      .setTextFile(textFile)
      .initialize()
      .loadTrain()
    lemmaLM
  }

  def load(): LemmaLM = {
    val lemmaLM = new LemmaLM(initTrain("lemmaBigLM"))
      .initialize()

    lemmaLM
  }

  def makeToken(token: String, lm: AbstractLM): String = {
    token.replaceAll(lm.lm.transducer.split, "")
  }


  def partitionLemma(lm: LemmaLM, params: Params): (String => String) = {

    val fn = (sentence: String) => {
      val tokens = tokenizer.standardTokenizer(sentence)
      lm.findMinSlideSplitSentence(tokens).mkString(" ")
    }
    fn
  }

  def partitionMasker(params: Params): (String => String) = {
    //val lm = train(params)
    val fn = (sentence: String) => {
      val sequence = tokenizer.standardTokenizer(sentence)
      sequence.mkString(" ")
    }

    fn
  }

  def noMaxTokenLength(sentence: String): Boolean = {
    val tokens = sentence.split("\\s+")
    val avgLength = tokens.map(_.length).sum / tokens.length
    avgLength < 20
  }

  def construct(lemmaLM: LemmaLM, sourceFilename: String, destinationFilename: String, start: Int = 0): this.type = {
    val f = new File(destinationFilename)
    if (!f.exists()) {
      val params = init("bigLemma", 20)
      val filestream = new FileOutputStream(destinationFilename, false);
      val corpusPrint = new PrintWriter(filestream)
      Source.fromFile(sourceFilename).getLines()
        .filter(noMaxTokenLength)
        .zipWithIndex.filter(pair => pair._2 >= start)
        .take(params.maxSentences)
        .foreach { case (paragraph, index) => {

          println(s"Processing paragraph index ${index}/${params.maxSentences}")
          var sentences = sentenceHMM.split(paragraph)
          sentences = if (sentences.isEmpty) sentenceHMM.candidateSplit(paragraph)
          else sentences

          //sentences = sentences.filter(sentence => noMaxTokenLength(sentence) && sentence.length < params.maxSentenceLength)
          sentences.par.map(sentence=> tokenizer.standardTokenizer(sentence))
            .filter(_.nonEmpty).map(tokens => {
            val lemmas = lemmaLM.findMinSlideSplitSentence(tokens)
              .flatMap(lemma=> lemma.split(lemmaLM.split))
            (tokens, lemmas)
          }).toArray.foreach {
            case (tokens, lemmas) => {
              corpusPrint.println(tokens.mkString(" "))
              corpusPrint.println(lemmas.mkString(" "))
              corpusPrint.flush()
            }
          }

        }
        }

      corpusPrint.close()
      System.gc()
    }

    this
  }

  def construct(sourceFilename: String, destinationFolder: String): this.type = {
    val p = init("bigLemma", 20)
    val lm = train(sourceFilename)
    //val lm = load()
    var start = 0

    new File(destinationFolder).mkdir()
    for (i <- 0 until p.epocs) {
      println("Epoc: " + i)
      construct(lm, sourceFilename, destinationFolder + "bigLemma-" + i + ".txt", start)
      start += p.maxSentences
      System.gc()
    }

    this
  }
}

object LMLemmaDataset {
  def main(args: Array[String]): Unit = {
    val srcFilename = "resources/text/sentences/sentences-april-v2-tr.txt"
    val dstFilename = "resources/text/lemma-april/"
    val dataset = new LMLemmaDataset()
    //val fn = dataset.partitionLemma(dataset.init("bigLemma", 20))
    //val result = fn("makam aynı makam para aynı para")
    //result.map(s => s).foreach(sentence => println(sentence))
    dataset.construct(srcFilename, dstFilename)
  }
}
