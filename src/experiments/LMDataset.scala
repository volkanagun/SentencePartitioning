package experiments

import io.netty.util.{Timeout, TimerTask}
import transducer.{AbstractLM, LMExperiment}
import utils.Tokenizer

import java.io.{File, PrintWriter}
import java.util
import java.util.Timer
import java.util.concurrent.{Callable, Executors, ForkJoinPool, TimeUnit}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport
import scala.io.Source
import scala.util.Random
import scala.util.control.Breaks


class LMDataset {

  case class KeyPair(w1: String, w2: String) {
    override def hashCode(): Int = w1.hashCode * 7 + w2.hashCode

    override def equals(obj: Any): Boolean = {
      val other = obj.asInstanceOf[KeyPair]
      other.hashCode() == hashCode()
    }
  }


  val mainFilename = "resources/text/sentences/sentences-april-v2-tr.txt"
  val tokenizer = new Tokenizer()


  def constructNER(): Unit = {
    construct("resources/evaluation/ner/train.txt", "resources/text/ner/main.txt")
  }

  def construct(trainFilename: String, targetFilename: String, wordCount: Int = 10): Unit = {
    var wordMap = Source.fromFile(trainFilename).getLines().flatMap(line => {
      line.split("[\\s\\p{Punct}]+")
    }).toSet.toArray

    println("Word count: " + wordMap.length)

    val sentences = Source.fromFile(mainFilename).getLines().toArray.filter(sentence => sentence.length < 200)
    val foundList = wordMap.zipWithIndex.par.flatMap(wordPair => {
      println("Word index: " + wordPair._2)
      val word = wordPair._1
      val selected = sentences.filter(sentence => sentence.contains(word))
        .take(wordCount)
      selected
    }).toArray

    val shuffle = Random.shuffle(foundList.toSeq)
    val pw = new PrintWriter(targetFilename)
    shuffle.foreach(line => pw.println(line))
    pw.close()
  }

  def checkValidWord(text: String, length: Int): Boolean = {
    !text.split("\\s+").exists(word => word.length > length)
  }

  def construct(lm:AbstractLM, taskName: String): Unit = {

    val params = lm.getParams
    val fname = new File(params.corpusFilename(taskName))
    val stats = new LMStats()

    println(s"Constructing corpus for ${taskName} and ${params.lmWindowLength}")
    println(s"Corpus filename: ${fname}")


    if (!fname.exists() || params.lmForceTrain) {
      val corpusPrint = new PrintWriter(params.corpusFilename(taskName))
      val sentenceSplit = (sentence:String) => lm.splitSentence(tokenizer.standardTokenizer(sentence))

      var index = 0
      var count = 0
      while (count < params.corpusEpocs) {
        Source.fromFile(params.mainCorpusFilename(taskName)).getLines()
          .filter(text => text.length < params.lmMaxSentenceLength)
          .filter(text => checkValidWord(text, params.lmTokenLength)).zipWithIndex.filter(_._2 >= index)
          .take(params.corpusMaxSentence)
          .map(_._1).toArray.sliding(params.lmThreads, params.lmThreads).foreach(sentences => {
            stats.start()
            val partitionedSentences = sentences.par.map(sentenceSplit)
            stats.end()
            val tokenCount = sentences.map(_.split("[\\s\\p{Punct}]").length).sum
            val sentenceCount = sentences.length
            stats.incTokenCount(tokenCount)
            stats.incSentenceCount(sentenceCount)

            partitionedSentences.toArray.foreach(tokens => {
              corpusPrint.println(tokens.mkString(" "))
              corpusPrint.flush()
            })
          })
        count += 1
        index += params.corpusMaxSentence
        println("Epoc: " + count + " / " + params.corpusEpocs)
        System.gc()
      }

      corpusPrint.close()
    }


    params.setStats(stats)
  }

/*

  def construct(): LMDataset = {
    val tasks = new Params().tasks

    tasks.foreach(task => construct(task))
    this
  }
*/
/*
  def constructParallel(): LMDataset = {
    val tasks = new Params().tasks
    tasks.foreach(task => constructParallel(task))
    this
  }*/
/*

  def densityParallel(): LMDataset = {
    val tasks = new Params().tasks
    tasks.foreach(task => densityParallel(task))
    this
  }
*/
/*

  def construct(taskName: String): LMDataset = {
    val p = new Params()
    val ranges = p.windows
    val models = p.adapters
    models.foreach(modelName => {
      ranges.foreach(windowSize => {
        val p = Params(modelName, windowSize)
        construct(p, taskName)
      })
    })
    this
  }

  def constructParallel(adapterName: String): LMDataset = {
    val p = new Params()
    val ranges = p.windows
    val models = p.adapters
    models.foreach(modelName => {
      ranges.par.foreach(windowSize => {
        val p = Params(modelName, windowSize)
        construct(p, adapterName)
      })
    })
    this
  }
*/
/*
  def densityParallel(adapterName: String): LMDataset = {
    val p = new Params()
    val ranges = p.windows
    val models = p.adapters
    models.foreach(modelName => {
      ranges.par.foreach(windowSize => {
        val p = Params(modelName, windowSize)
        densityScores(p, adapterName)
      })
    })
    this
  }*/
/*

  def partition(params: Params): (String => Array[String]) = {
    if ("lm-syllable".equals(params.adapterName)) partitionSlide(params)
    else if ("lm-skip".equals(params.adapterName)) partitionSlide(params)
    else if ("lm-rank".equals(params.adapterName)) partitionSlide(params)
    else if ("lm-rank-efficient".equals(params.adapterName)) partitionEfficient(params)
    else if ("lm-lemma".equals(params.adapterName)) partitionSlide(params)
    else if ("lm-ngram".equals(params.adapterName)) partitionSlide(params)
    else if ("frequent-ngram".equals(params.adapterName)) partitionSlide(params)
    else null;
  }

  def partitionFunc(params: Params): (String => Array[Array[String]]) = {

    if ("lm-syllable".equals(params.adapterName)) partitionFunction(params)
    else if ("lm-skip".equals(params.adapterName)) partitionFunction(params)
    else if ("lm-rank".equals(params.adapterName)) partitionFunction(params)
    else if ("lm-lemma".equals(params.adapterName)) partitionFunction(params)
    else if ("lm-ngram".equals(params.adapterName)) partitionFunction(params)
    else if ("frequent-ngram".equals(params.adapterName)) partitionFunction(params)
    else null;

  }
*/

/*

  def partitionSlide(params: Params): (String => Array[String]) = {

    val lm = params.model(params, params.adapterName)
    val fn = (sentence: String) => {

      val sequence = tokenizer.standardTokenizer(sentence)

      sequence.sliding(lm.params.lmWindowLength, lm.params.lmWindowLength).zipWithIndex.toArray.par.map(pairs => {
          val tokens = pairs._1
          val task = new Callable[Array[String]]() {
            def call(): Array[String] = {
              lm.splitSentence(tokens)
            }
          }
          (pairs._2, task.call())
        }).toArray.sortBy(_._1)
        .flatMap(_._2)
    }


    fn
  }*/
/*
  def partitionEfficient(params: Params): (String => Array[String]) = {

    val lm = params.model(params, params.adapterName)
    val fn = (sentence: String) => {
      val sequence = tokenizer.standardTokenizer(sentence)
      val result = lm.splitSentence(sequence)
      result
    }


    fn
  }

  def partitionFunction(params: Params): (String => Array[Array[String]]) = {
    val lm = params.modelDefault(params.adapterName, params.lmWindowLength)
    val fn = (sentence: String) => {
      val sequence = tokenizer.standardTokenizer(sentence)
      val newSentence = lm.splitSentence(sequence)
      val words = newSentence.map(token => token.split(lm.lm.seqTransducer.split))
      words
    }

    fn
  }*/

  def densityScores(params: Params, taskName: String): Unit = {
    val densityFile = new File(params.densityFilename(taskName))
    val corpusFile = new File(params.corpusFilename(taskName))
    val rnd = new Random(7)
    if (corpusFile.exists()) {


      val tokens = Source.fromFile(corpusFile).getLines()
        .toArray
        .map(line => line.split("\\s+"))

      val samples = rnd.shuffle(tokens.toSeq)
        .take(10000)

      println("Constructing density file:" + densityFile.getPath)
      val pairs = samples.map(sentence => {
        sentence.flatMap(token => {
          sentence.map(other => KeyPair(token, other))
      })}).flatten.groupBy(pair => pair).view.mapValues(pairs => pairs.length.toDouble).toMap

      val totalEdges = pairs.map(_._2).sum

      val distinctTokens = samples.flatMap(items => items).toSet.size
      val totalTokens = samples.flatMap(items => items).length
      val totalSentences = samples.length
      val graphDensity = 2 * totalEdges / (distinctTokens * (distinctTokens - 1))

      val pw = new PrintWriter(densityFile)
      pw.println("Window size: " + params.lmWindowLength + ", Task: " + taskName + ", Model: " + params.adapterName)
      pw.println("Distinct token size: " + distinctTokens)
      pw.println("Average distinct token size: " + distinctTokens.toDouble/totalSentences)
      pw.println("Total token size: " + totalTokens)
      pw.println("Total sentences: " + totalSentences)
      pw.println("Average sentence size: " + totalTokens.toDouble / totalSentences)
      pw.println("Total edge size: " + totalEdges)

      pw.println("Graph density: " + graphDensity)
      pw.close()
    }
  }
}
