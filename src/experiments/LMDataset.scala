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

  val mainFilename = "resources/text/sentences/sentences-april-v2-tr.txt"
  val experimentLM = new LMExperiment()
  val tokenizer = new Tokenizer()

  def train(): LMDataset = {
    experimentLM.train()
    this
  }

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

  def construct(params: Params, taskName: String): Unit = {
    println(s"Constructing corpus for ${taskName} and ${params.lmWindowLength}")
    val fname = new File(params.corpusFilename(taskName))
    val stats = new LMStats()

    if (!fname.exists() || params.lmForceTrain) {
      val corpusPrint = new PrintWriter(params.corpusFilename(taskName))
      val fn = partition(params)

      var index = 0
      var count = 0
      while (count < params.corpusEpocs) {
        Source.fromFile(params.mainCorpusFilename(taskName)).getLines()
          .filter(text => text.length < params.lmMaxSentenceLength)
          .filter(text => checkValidWord(text, params.lmTokenLength)).zipWithIndex.filter(_._2 >= index)
          .take(params.corpusMaxSentence)
          .map(_._1).toArray.sliding(params.lmThreads, params.lmThreads).foreach(sentences => {
            stats.start()
            val partitionedSentences = sentences.par.map(fn)
            stats.end()
            val tokenCount =  sentences.map(_.split("[\\s\\p{Punct}]").length).sum
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


  def construct(): LMDataset = {
    val tasks = new Params().tasks

    tasks.foreach(task => construct(task))
    this
  }

  def constructParallel(): LMDataset = {
    val tasks = new Params().tasks
    tasks.foreach(task => constructParallel(task))
    this
  }

  def construct(taskName: String): LMDataset = {
    val p = new Params()
    val ranges = p.windows
    val models = p.adapters
    models.foreach(modelName => {
      ranges.foreach(windowSize => {
        val p = experimentLM.init(modelName, windowSize)
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
        val p = experimentLM.init(modelName, windowSize)
        construct(p, adapterName)
      })
    })
    this
  }

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

  def makeToken(token: String, lm: AbstractLM): String = {
    token.replaceAll(lm.lm.transducer.split, "")
  }


  def executeTask(task: Callable[Array[String]], time: Long): Array[String] = {
    val executor = Executors.newSingleThreadExecutor()
    val future = executor.submit(task)
    var words = Array[String]()
    try {
      words = future.get(time, TimeUnit.SECONDS)
    }
    catch {
      case _: Exception => println("Error in 50 seconds...")
    }
    finally {
      executor.shutdown()
    }

    words
  }

  def partitionSlide(params: Params): (String => Array[String]) = {

    val lm = experimentLM.model(params, params.adapterName)
    val fn = (sentence: String) => {

      val sequence = tokenizer.standardTokenizer(sentence)

      sequence.sliding(lm.params.lmWindowLength, lm.params.lmWindowLength).zipWithIndex.toArray.par.map(pairs => {
          val tokens = pairs._1
          val task = new Callable[Array[String]]() {
            def call(): Array[String] = {
              lm.findMinSplitSentence(tokens)
                .head.split(lm.splitSpace)
            }
          }
          (pairs._2, task.call())
        }).toArray.sortBy(_._1)
        .flatMap(_._2)


    }


    fn
  }

  def partitionEfficient(params: Params): (String => Array[String]) = {

    val lm = experimentLM.model(params, params.adapterName)
    val fn = (sentence: String) => {
      val sequence = tokenizer.standardTokenizer(sentence)
      sequence.sliding(lm.params.lmWindowLength, lm.params.lmWindowLength).zipWithIndex.toArray.flatMap(pairs => {
        val tokens = pairs._1
        val result = lm.findMinSplitEfficient(tokens).head
        result.split(lm.split)
      })
    }


    fn
  }

  def partitionFunction(params: Params): (String => Array[Array[String]]) = {
    val lm = experimentLM.model(params.adapterName, params.lmWindowLength)
    val fn = (sentence: String) => {
      val sequence = tokenizer.standardTokenizer(sentence)
      val newSentence = lm.findMinSplitSentence(sequence).head
      val words = newSentence.split("\\s+").map(token => token.split(lm.lm.seqTransducer.split))
      words
    }

    fn
  }
}

object LMDataset extends LMDataset() {

  def main(args: Array[String]): Unit = {
    //constructNER()
    //train()
    constructParallel()
  }
}