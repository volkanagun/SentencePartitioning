package transducer

import experiments.Params

import java.io._
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import java.util.{Locale, Random}
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable}
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future, Promise, TimeoutException}
import scala.io.Source
import scala.util.{Failure, Success}


object TransducerOp {

  var locale = new Locale("tr")


  def test(): Unit = {
    val transducer = new Transducer()


    transducer.addPrefix(Array("elma"))
    transducer.addPrefix("elmalardan")
    transducer.addPrefix("elmalar")


    val input0 = "elmalardan".toCharArray.map(_.toString)
    val input5 = "ları".toCharArray.map(_.toString)
    /*val result1 = transducer.search("halklar")
    val result2 = transducer.search("halk")
    val result3 = transducer.search("halkdan")
    val result4 = transducer.search("halka")*/
    val result0 = transducer.multipleSearch(input0)
    val result1 = transducer.longestSearch("almanlar")
    val result5 = transducer.multipleSearch(input5)

    /*println("halklar:" + result1)
    println("halk:" + result2)
    println("halkdan:" + result3)
    println("halka:" + result4)*/
    println("Multiple:" + result0.mkString("\n"))
    println("Longest:" + result1.mkString(""))
    println("Multiple:" + result5.mkString("\n"))
  }

  def stemPartition(token: String): Array[Array[String]] = stemPartition(token, 5)

  def stemPartition(token: String, stemEnd: Int = 1): Array[Array[String]] = {

    var array = Array[Array[String]]()
    val min = Math.min(stemEnd, token.length)
    for (i <- min until token.length) {
      array = array :+ Array(token.substring(0, i), token.substring(i))
    }

    array = array :+ Array(token)
    array
  }


  def fromBinary(modelFilename: => String): Transducer = {
    if (new File(modelFilename).exists()) {
      println("Loading from " + modelFilename)
      val transducer = new Transducer()
      val inputStream = new FileInputStream(modelFilename)
      transducer.load(new ObjectInputStream(inputStream))
      inputStream.close()
      transducer
    }
    else {
      new Transducer()
    }
  }


  def fromText(transducer: Transducer, filename: => String, params: Params): Transducer = {

    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text")
      val wordTokenizer = new WordTokenizer()
      val ranges = Range(0, params.lmEpocs).toArray

      val results = ranges.sliding(params.lmThreads, params.lmThreads).foreach(epocs => {
        epocs.par.map(epoc => {
          println("Dictionary epoc: " + epoc + "/" + ranges.length)
          val crr = new Transducer()
          val index = epoc * params.lmMaxSentence
          Source.fromFile(filename).getLines()
            .zipWithIndex
            .filter(pair => pair._2 >= index && pair._1.length < params.lmMaxSentenceLength)
            .take(params.lmMaxSentence)
            .map(pair => pair._1).toArray
            .map(sentence => wordTokenizer.standardTokenizer(sentence.toLowerCase(locale)))
            .map(tokens => tokens.filter(item => item.matches("\\p{L}+")).map(_.trim))
            .map(tokens => tokens.flatMap(token => stemPartition(token, params.lmStemLength)))
            .toArray
            .flatMap(sentences => sentences)
            .foreach(sequence => {
              sequence.foreach(token => crr.addPrefix(token))
            })
          crr
        }).toArray.foreach(crr => {

          println("Merging transducer...")
          transducer.merge(crr)
        })

        System.gc()
      })

      println("Constructing dictionary from text is finished.")
      transducer
    }
    else {
      transducer
    }
  }

  def fromMorphology(transducer: Transducer, filename: => String, params: Params): Transducer = {

    if (params.lmTrainDictionary) {
      println("Constructing dictionary from morphological text")
      val wordTokenizer = new WordTokenizer()
      val rangeMax = params.lmrange * params.lmEpocs
      val random = new Random(42)
      val ranges = Range(0, params.lmEpocs)
      val lines = Source.fromFile(filename).getLines().take(rangeMax).toArray
        .zipWithIndex

      ranges.par.map(epoc => {
        val crr = new Transducer()
        println("Dictionary epoc: " + epoc + "/" + ranges.length)
        val randomSet = Range(0, params.lmMaxSentence)
          .map(_ => random.nextInt(rangeMax))
          .toSet

        lines.filter(pair => randomSet.contains(pair._2))
          .map(pair => pair._1)
          .take(params.lmMaxSentence)
          .par
          .map(sentence => sentence.split("\\s"))
          .toArray
          .foreach(sequence => {
            sequence.foreach(token => {
              val inputOutput = token.split("\\|").map(item => item.split("\\:"))
              val input = inputOutput.map(item => item.head)
              val output = inputOutput.map(item => item.last)
              crr.addInput(input, output)
            })
          })
        crr
      }).toArray.foldRight(transducer) { case (a, main) => main.merge(a) }
    }

    transducer
  }

  def fromTextByInfer(inference: String => Array[String], transducer: Transducer, filename: => String, params: Params): Transducer = {

    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text")
      val random = new Random()
      val wordTokenizer = new WordTokenizer()

      val ranges = Range(0, params.lmEpocs)

      ranges.sliding(params.lmThreads, params.lmThreads).foreach(epocs => {
        epocs.par.map(epoc => {
          val crr = new Transducer()
          println(s"Epoc: ${epoc} for ${params.adapterName}")
          val start = epoc * params.lmMaxSentence
          Source.fromFile(filename).getLines().filter(_.length <= params.lmMaxSentenceLength)
            .zipWithIndex.filter(pair => pair._2 >= start)
            .map(pair => pair._1)
            .take(params.lmMaxSentence)
            .map(line => wordTokenizer.standardTokenizer(line).filter(item => item.matches("\\p{L}+") && item.length < params.lmTokenLength))
            .flatMap(tokens => tokens)
            .map(_.trim)
            .map(sequence => {

              inference(sequence)
            })
            .foreach(subseq => {
              crr.addPrefix(subseq)
            })
          crr
        }).toArray.foreach(crr => {
          println("Merging...")
          transducer.merge(crr)
        }
        )

        System.gc()

      })


      println("Constructing dictionary from text is finished.")

    }

    transducer
  }

  def fromDictionary(modelFilename: String, filename: => String, params: Params): Transducer = {


    val transducer = fromBinary(modelFilename)
    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text lexicon...")
      Source.fromFile(filename).getLines()
        .toArray
        .par.map(line => line.split("\t").head.toLowerCase(locale)
          .replaceAll("[\\&\\_]", "")
          .replaceAll("([au])([bcçdfgğhjklmnprsştvyz])E$", "$1$2a")
          .replaceAll("([eü])([bcçdfgğhjklmnprsştvyz])E$", "$1$2e").toLowerCase(locale).trim)
        .toArray.foreach(token => {
          transducer.addPrefix(token)
        })


    }

    transducer
  }

  def fromDictionary(transducer: Transducer, modelFilename: => String, filename: => String, params: Params): Transducer = {

    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text lexicon...")
      Source.fromFile(filename).getLines()
        .map(line => line.split("\t").head.toLowerCase(locale)
          .replaceAll("[\\~\\&\\_]", "")
          .replaceAll("([au])([bcçdfgğhjklmnprsştvyz])E$", "$1$2a")
          .replaceAll("([eü])([bcçdfgğhjklmnprsştvyz])E$", "$1$2e").toLowerCase(locale).trim)
        .foreach(token => {
          transducer.addPrefix(token)
        })


      println("Constructing dictionary from text lexicon finished")
    }

    transducer
  }

  def fromDictionaryByInfer(inference: String => Array[String], transducer: Transducer, filename: => String, params: Params): Transducer = {
    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text lexicon...")
      val array = Source.fromFile(filename).getLines().toArray.par
        .map(line => line.split("\t").head).toArray

      array.flatMap(token => {
          token.replaceAll("[\\~\\&\\_]", "")
            .replaceAll("([au])([bcçdfgğhjklmnprsştvyz])E$", "$1$2a")
            .replaceAll("([eü])([bcçdfgğhjklmnprsştvyz])E$", "$1$2e")
            .toLowerCase(locale).trim.split("\\s")
        }).par.map(token => inference(token).take(params.lmCandidateCount))
        .toArray
        .foreach(subsequence => {
          subsequence.foreach(syllable => {
            transducer.addPrefix(syllable.split(transducer.split))
          })
        })
      println("Constructing dictionary from text lexicon finished")
    }

    transducer
  }


  def fromSyllables(): Transducer = {

    val transducer = new Transducer()
    var vowels = Array("a", "e", "ı", "i", "o", "ö", "u", "ü")
    var consonants = Array("b", "c", "ç", "d", "f", "g", "ğ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "ş", "t", "v", "w", "y", "q", "z", "x")

    val v = vowels
    val vc = vowels.flatMap(v => consonants.map(c => v + c))
    val cv = vowels.flatMap(v => consonants.map(c => c + v))
    val cvc = consonants.flatMap(c1 => vowels.flatMap(v => consonants.map(c2 => c1 + v + c2)))
    val vcc = vowels.flatMap(v => consonants.flatMap(c1 => consonants.map(c2 => v + c1 + c2)))
    val ccv = consonants.flatMap(c1 => consonants.flatMap(c2 => vowels.map(v => c1 + c2 + v)))
    val cvcc = consonants.flatMap(c0 => consonants.flatMap(c1 => consonants.flatMap(c2 => vowels.map(v => c0 + v + c1 + c2))))
    val ccvc = consonants.flatMap(c0 => consonants.flatMap(c1 => consonants.flatMap(c2 => vowels.map(v => c0 + c1 + v + c2))))

    val all = vc ++ cv ++ cvc ++ vcc ++ ccv ++ cvcc ++ v
    all.foreach(item => transducer.addPrefix(item))
    transducer
  }

  def syllableNotAccept(): Array[String] = {
    var vowels = Array("a", "e", "ı", "i", "o", "ö", "u", "ü").mkString("[", "", "]")
    var consonants = Array("b", "c", "ç", "d", "f", "g", "ğ", "h", "j", "k", "l", "m", "n", "p", "r", "s", "ş", "t", "v", "w", "y", "z", "x").mkString("[", "", "]")
    var split = "[\\#\\$]"
    var boundary = "^"
    val item1 = split + consonants + split
    val item2 = split + vowels + split
    val item3 = split + consonants + consonants + split
    val item4 = split + consonants + consonants + split
    val item5 = consonants + split + vowels

    Array(item1, item2, item3, item4, item5)

  }


  def runWithTimeout[T](operation: => T, timeout: Duration): Unit = {
    implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global
    val promise = Promise[T]()

    // Wrap the operation in a Future
    val future = Future {
      val result = operation
      promise.success(result)
    }


    try {
      Await.ready(promise.future, timeout)
      promise.future.value.get match {
        case Success(value) => Right(value)
        case Failure(exception) => Left(exception.getMessage)
      }
    } catch {
      case e: TimeoutException => println("Operation timed out")
      case e: Exception => println(s"Operation failed with exception: ${e.getMessage}")
    }
  }

  def train(transducer: Transducer, filename: => String, modelFilename: => String, params: Params): TransducerLM = {

    var transducerLM = loadLM(modelFilename, transducer)
    var index = 0
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i)
      transducerLM = train(transducerLM, filename, params, index)
      transducerLM.prune().normalize()
      saveLM(modelFilename, transducerLM)
      index += params.lmMaxSentence
    }

    transducerLM
  }

  def trainCombinatoric(transducer: Transducer,
                        filename: => String, modelFilename: => String, params: Params): TransducerLM = {

    var transducerLM = loadLM(modelFilename, transducer)
    var start: Int = 0
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i)
      transducerLM = trainCombinatoric(transducerLM, filename, params, start)
      transducerLM.prune().normalize()
      saveLM(modelFilename, transducerLM)
      start += params.lmMaxSentence
    }

    transducerLM
  }

  def trainParallel(transducer: Transducer, filename: => String, modelFilename: => String, params: Params): TransducerLM = {

    var transducerLM = loadLM(modelFilename, transducer)
    var index = 0
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i)
      transducerLM = trainParallel(transducerLM, filename, params, index)
      transducerLM.prune().normalize()
      saveLM(modelFilename, transducerLM)
      index += params.lmMaxSentence
    }

    transducerLM
  }

  def trainParallelCombinatoricBySlide(transducer: Transducer, filename: => String, modelFilename: => String, params: Params): TransducerLM = {

    var transducerLM = loadLM(modelFilename, transducer)
    var start = 0
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i + " for " + params.adapterName)
      transducerLM = trainSlideParallelCombinatoric(transducerLM, filename, params, start)
      //transducerLM = trainSlideEfficientCombinatoric(transducerLM, filename, params, start)
      transducerLM.prune().normalize()
      saveLM(modelFilename, transducerLM)
      start += params.lmMaxSentence
    }

    transducerLM
  }

  def trainParallelCombinatoricBySkip(transducer: Transducer, filename: String, modelFilename: String, params: Params): TransducerLM = {

    var transducerLM = loadLM(modelFilename, transducer)
    var index = 0
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i + " for " + params.adapterName)
      //transducerLM = trainSkipEfficientCombinatoric(transducerLM, filename, params, index)
      transducerLM = trainSkipParallelCombinatoric(transducerLM, filename, params, index)
      transducerLM.prune().normalize()
      saveLM(modelFilename, transducerLM)
      index += params.lmMaxSentence
    }

    transducerLM
  }


  def trainParallelCombinatoricBySkip(transducer: Transducer, partitionFunc: (Array[String]) => Array[Array[String]],
                                      filename: String,
                                      modelFilename: String,
                                      params: Params): TransducerLM = {

    var transducerLM = loadLM(modelFilename, transducer)
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i + " for " + params.adapterName)
      val start = i * params.lmMaxSentence
      transducerLM = trainSkipCombinatoric(transducerLM, partitionFunc, filename, params, start)
      transducerLM.prune().normalize()
      saveLM(modelFilename, transducerLM)
    }

    transducerLM
  }

  def train(transducerLM: TransducerLM, filename: String, modelFilename: String, params: Params): TransducerLM = {

    var start = 0
    var newLM = train(transducerLM, filename, params, start)
    for (i <- 0 until params.lmEpocs) {
      println("Epocs: " + i + " for " + params.adapterName)
      start += params.lmMaxSentence
      newLM = train(newLM, filename, params, start)
      newLM.prune().normalize()
      saveLM(modelFilename, newLM)
    }

    newLM
  }

  def trainCombinatoric(transducerLM: TransducerLM,
                        filename: => String, modelFilename: => String,
                        params: Params): TransducerLM = {

    var start = 0
    var newLM = trainCombinatoric(transducerLM, filename, params, start)
    for (i <- 0 until params.lmEpocs) {
      println("Epoc: " + i + " for " + params.adapterName)
      start += params.lmMaxSentence
      newLM = trainCombinatoric(newLM, filename, params, start)
      newLM.prune().normalize()
      saveLM(modelFilename, newLM)
    }

    newLM
  }

  def trainParallel(transducerLM: TransducerLM, filename: => String, modelFilename: => String, params: Params): TransducerLM = {

    val range = Range(0, params.lmEpocs).toArray

    range.sliding(params.lmThreads, params.lmThreads).foreach(iset => {
      iset.par.map(i => {
        println(s"Epoc: ${i} for ${params.adapterName}")
        val start = i * params.lmMaxSentence
        val lmCopy = new TransducerLM(transducerLM.transducer)
        trainParallel(lmCopy, filename, params, start)
      }).toArray.foreach(crrLM => transducerLM.merge(crrLM))

      System.gc()
    })

    transducerLM.prune().normalize()
    saveLM(modelFilename, transducerLM)
    transducerLM
  }


  def trainParallelCombinatoricBySlide(transducerLM: TransducerLM, filename: String, modelFilename: String, params: Params): TransducerLM = {

    val range = Range(0, params.lmEpocs)
    val nsize = params.lmThreads

    range.sliding(nsize, nsize).foreach(iset => {
      iset.par.map(i => {
        println(s"Epoc: ${i} for ${params.adapterName}")
        val index = i * params.lmMaxSentence
        val crrLM = trainSlideParallelCombinatoric(new TransducerLM(transducerLM.transducer), filename, params, index)
        crrLM
      }).toArray.foreach(crrLM =>{
        transducerLM.merge(crrLM)
        System.gc()
      })
    })

    transducerLM
      .prune()
      .normalize()


    saveLM(modelFilename, transducerLM)
    transducerLM
  }

  def trainParallelCombinatoricBySkip(transducerLM: TransducerLM, filename: => String,
                                      modelFilename: => String,
                                      params: Params): TransducerLM = {

    val range = Range(0, params.lmEpocs)
    val threads = params.lmThreads

    range.sliding(threads, threads).foreach(iset => {

      iset.par.map(i => {
        println(s"Epoc: ${i}/${params.lmEpocs} for ${params.adapterName}")
        val index = i * params.lmMaxSentence
        val lm = trainSkipParallelCombinatoric(new TransducerLM(transducerLM.transducer), filename, params, index)
        lm
      }).toArray.foreach(lm => {
        transducerLM.mergeSequence(lm)
        System.gc()
      })


    })

    transducerLM
      .prune()
      .normalize()

    saveLM(modelFilename, transducerLM)
    transducerLM


  }

  def trainEfficientCombinatoricBySkip(transducerLM: TransducerLM, filename: => String,
                                       modelFilename: => String,
                                       params: Params): TransducerLM = {

    val range = Range(0, params.lmEpocs)
    val threads = params.lmThreads
    range.sliding(threads, threads).foreach(iset => {

      iset.par.map(i => {
        println(s"Epoc: ${i}/${params.lmEpocs} for ${params.adapterName}")
        val index = i * params.lmMaxSentence
        val lm = trainSkipEfficientCombinatoric(new TransducerLM(transducerLM.transducer), filename, params, index)
        lm
      }).toArray.foreach(lm => {
        transducerLM.merge(lm)
        System.gc()
      })


    })

    transducerLM
      .prune(params.lmPrune)
      .normalize()

    saveLM(modelFilename, transducerLM)
    transducerLM


  }

  def trainParallelCombinatoricBySkip(transducerLM: TransducerLM, partition: (Array[String] => Array[Array[String]]), filename: String,
                                      modelFilename: String,
                                      params: Params): TransducerLM = {

    val range = Range(0, params.lmEpocs)

    range.sliding(params.lmThreads, params.lmThreads).foreach(iset => {
      iset.par.map(i => {
        val start = i * params.lmMaxSentence
        println(s"Epoc: ${i}/${params.lmEpocs} for ${params.adapterName}")
        val lm = trainSkipCombinatoric(new TransducerLM(transducerLM.transducer), partition, filename, params, start)
        lm
      }).toArray.foreach(lm => {
        transducerLM.merge(lm)
        System.gc()
      })
    })

    transducerLM
      .prune()
      .normalize()

    saveLM(modelFilename, transducerLM)
    transducerLM
  }

  def trainEfficientCombinatoricBySkip(transducerLM: TransducerLM, partition: (Array[String] => Array[Array[String]]), filename: String,
                                       modelFilename: String,
                                       params: Params): TransducerLM = {

    val range = Range(0, params.lmEpocs)
    range.sliding(params.lmThreads, params.lmThreads).foreach(iset => {
      iset.par.map(i => {
        val start = i * params.lmMaxSentence
        println(s"Epoc: ${i}/${params.lmEpocs} for ${params.adapterName}")
        val lm = trainSkipEfficientCombinatoric(new TransducerLM(transducerLM.transducer), partition, filename, params, start)
        lm
      }).toArray.foreach(lm => {
        transducerLM.merge(lm)
        System.gc()
      })
    })

    transducerLM
      .normalize()

    saveLM(modelFilename, transducerLM)
    transducerLM
  }

  def train(transducer: Transducer, filename: => String, params: Params, start: Int): TransducerLM = {

    var cnt = 0;
    val transducerLM = new TransducerLM(transducer)
    val random = new Random()
    Source.fromFile(filename).getLines().to(LazyList).zipWithIndex
      .filter(pair => pair._2 >= start)
      .map(_._1)
      .foreach {
        line => {
          val arr = line.split("[\\s\\p{Punct}]+")
            .map(_.toLowerCase(locale))
            .filter(item => item.matches("\\p{L}+"))
            .filter(item => item.length < params.lmTokenLength)

          println("Inferring line:" + cnt)
          transducerLM.count(arr, params.lmTopSplit, 3)

        }
      }

    transducerLM.prune().normalize()
  }

  def trainCombinatoric(transducer: Transducer, filename: => String, params: Params, start: Int): TransducerLM = {

    val transducerLM = new TransducerLM(transducer)


    Source.fromFile(filename).getLines().to(LazyList).zipWithIndex
      .filter(pair => pair._2 >= start)
      .map(_._1)
      .foreach {
        line => {
          var arr = line.split("[\\s\\p{Punct}]+")
            .map(_.toLowerCase(locale))
            .filter(item => item.matches("\\p{L}+"))
            .filter(item => item.length < params.lmTokenLength)
            .sliding(params.lmWindowLength, 1)
            .toArray


          arr.foreach(sequence => transducerLM.countCombinatoric(sequence, params.lmTopSplit, params.lmSlideLength))

        }
      }

    transducerLM.prune().normalize()
  }

  def train(transducerLM: TransducerLM, filename: => String, params: Params, start: Int): TransducerLM = {


    Source.fromFile(filename).getLines().zipWithIndex.filter(_._2 >= start)
      .take(params.lmMaxSentence).map(_._1)
      .foreach {
        line => {
          val arr = line.split("[\\s\\p{Punct}]+")
            .map(_.toLowerCase(locale))
            .filter(item => item.matches("\\p{L}+"))
            .filter(item => item.length < params.lmTokenLength)
          transducerLM.count(arr, params.lmTopSplit, params.lmSlideLength)

        }
      }

    transducerLM.prune().normalize()

  }

  def trainCombinatoric(transducerLM: TransducerLM, filename: => String, params: Params, start: Int): TransducerLM = {

    Source.fromFile(filename).getLines().to(LazyList).zipWithIndex
      .filter(pair => pair._2 >= start)
      .map(_._1)
      .foreach {
        line => {
          val arr = line.split("[\\s\\p{Punct}]+")
            .map(_.toLowerCase(locale))
            .filter(item => item.matches("\\p{L}+"))
            .filter(item => item.length < params.lmTokenLength)
            .sliding(params.lmWindowLength, 1)
            .toArray
          arr.foreach(sequence => transducerLM.countCombinatoric(sequence, params.lmTopSplit, params.lmSlideLength))

        }
      }

    transducerLM.prune().normalize()

  }

  def trainParallel(transducerLM: TransducerLM, filename: => String, params: Params, start: Int): TransducerLM = {

    Source.fromFile(filename).getLines()
      .zipWithIndex.filter(_._2 >= start)
      .take(params.lmMaxSentence)
      .map(_._1)
      .foreach {
        line => {

          val arr = line.split("[\\s\\p{Punct}]+")
            .map(_.toLowerCase(locale))
            .filter(item => item.matches("\\p{L}+"))
            .filter(item => item.length < params.lmTokenLength)
            .sliding(params.lmWindowLength)

          arr.toArray.foreach(transducerLM.countCombinatoric(_, params.lmTopSplit, params.lmSlideLength))
        }
      }


    transducerLM

  }

  def trainSlideParallelCombinatoric(transducerLM: TransducerLM, filename: String, params: Params, start: Int): TransducerLM = {


    Source.fromFile(filename).getLines()
      .filter(sentence => sentence.length < params.lmMaxSentenceLength)
      .zipWithIndex.filter(_._2 >= start).take(params.lmMaxSentence).map(_._1)
      .toArray
      .foreach(line => {
        line.split("[\\s\\p{Punct}]+")
          .map(_.toLowerCase(locale))
          .filter(item => item.matches("\\p{L}+"))
          .filter(item => item.length < params.lmTokenLength)
          .sliding(params.lmWindowLength, 1)
          .foreach(sequence => transducerLM.countEfficient(sequence, params.lmSample,params.lmTopSplit, params.lmSlideLength, params.lmSkip))
      })

    transducerLM
  }

  /*
    def trainSlideEfficientCombinatoric(transducerLM: TransducerLM, filename: String, params: Params, start: Int): TransducerLM = {

      Source.fromFile(filename).getLines()
        .filter(sentence => sentence.length < params.lmMaxSentenceLength)
        .zipWithIndex.filter(_._2 >= start).take(params.lmMaxSentence).map(_._1)
        .toArray
        .foreach(line => {
          println(line)
          line.split("[\\s\\p{Punct}]+")
            .map(_.toLowerCase(locale))
            .filter(item => item.matches("\\p{L}+"))
            .filter(item => item.length < params.lmTokenLength)
            .sliding(params.lmWindowLength, 1)
            .toArray.par
            .map(seq => {
              val crrLM = transducerLM.copy()
              crrLM.countCombinatoric(seq, params.lmTopSplit, params.lmSlideLength)
              crrLM
            }).toArray.foreach(crrLM=> transducerLM.merge(crrLM))
        })

      transducerLM
    }
  */
  def trainSkipParallelCombinatoric(transducerLM: TransducerLM, filename: => String, params: Params, start: Int): TransducerLM = {

    var cnt = 0;
    val random = new Random()

    Source.fromFile(filename).getLines().filter(line => line.length < params.lmMaxSentenceLength)
      .zipWithIndex.filter(_._2 >= start)
      .map(_._1).take(params.lmMaxSentence)
      .toArray
      .foreach { line => {
        line.split("[\\s\\p{Punct}]+")
          .map(_.toLowerCase(locale))
          .filter(item => item.matches("\\p{L}+"))
          .filter(item => item.length < params.lmTokenLength)
          .sliding(params.lmWindowLength, 1)
          .toArray
          .foreach(sequence => transducerLM.countCombinatoric(sequence, params.lmSample, params.lmTopSplit, params.lmSlideLength, params.lmSkip))
      }
      }

    transducerLM
  }

  def computeWithTimeout(computation: () => Unit, timeoutSeconds: Int): Unit = {
    try {
      implicit val ec: ExecutionContext = ExecutionContext.global
      val future: Future[Unit] = Future {
        computation()
      }
      Await.ready(future, Duration(timeoutSeconds, TimeUnit.SECONDS))
    }
    catch {
      case ex: Exception => println("Timeout...")
    }
  }

  def trainSkipEfficientCombinatoric(transducerLM: TransducerLM, filename: => String, params: Params, start: Int): TransducerLM = {

    var cnt = 0;
    val random = new Random()

    Source.fromFile(filename).getLines().filter(line => line.length < params.lmMaxSentenceLength)
      .zipWithIndex.filter(_._2 >= start)
      .map(_._1).take(params.lmMaxSentence)
      .toArray
      .foreach { line => {
        line.split("[\\s\\p{Punct}]+")
          .map(_.toLowerCase(locale))
          .filter(item => item.matches("\\p{L}+"))
          .filter(item => item.length < params.lmTokenLength)
          .sliding(params.lmWindowLength, 1)
          .toArray
          .foreach(sequence => {
            val func = () => {
              transducerLM.countEfficient(sequence, params.lmSample, params.lmTopSplit, params.lmSlideLength, params.lmSkip)
            }
            func()
            //computeWithTimeout(func, params.lmMaxWaitSeconds)
          })
      }
      }

    transducerLM
  }

  /*def trainSkipEfficientCombinatoric(transducerLM: TransducerLM, filename: => String, params: Params, start: Int): TransducerLM = {

    var cnt = 0;
    val random = new Random()

    Source.fromFile(filename).getLines().filter(line => line.length < params.lmMaxSentenceLength)
      .zipWithIndex.filter(_._2 >= start)
      .map(_._1).take(params.lmMaxSentence)
      .toArray.par.map(line=>{
        val crrLM = transducerLM.copy()
        line.split("[\\s\\p{Punct}]+")
          .map(_.toLowerCase(locale))
          .filter(item => item.matches("\\p{L}+"))
          .filter(item => item.length < params.lmTokenLength)
          .sliding(params.lmWindowLength, 1)
          .toArray.foreach(seq=>{
            crrLM.countCombinatoric(seq, params.lmTopSplit, params.lmSlideLength, params.lmSkip)
          })
         crrLM
      }).toArray.foreach(crrLM => transducerLM.merge(crrLM))

    transducerLM
  }*/

  def trainSkipCombinatoric(transducerLM: TransducerLM, partitionFunc: (Array[String]) => Array[Array[String]], filename: => String, params: Params, start: Int): TransducerLM = {

    Source.fromFile(filename).getLines().to(LazyList).map(line => line.trim)
      .filter(line => line.nonEmpty && line.length > params.lmMinSentenceLength && line.length < params.lmMaxSentenceLength)
      .zipWithIndex.filter(pair => pair._2 >= start)
      .take(params.lmMaxSentence).map(_._1)
      .foreach { line => {

        line.split("[\\s\\p{Punct}]+")
          .map(_.toLowerCase(locale))
          .filter(item => item.matches("\\p{L}+"))
          .filter(item => item.length < params.lmTokenLength)
          .sliding(params.lmWindowLength, 1)
          .toArray
          .foreach(sequence => {
            transducerLM.countEfficientCombinatoric(sequence, partitionFunc, params.lmSample, params.lmWindowLength, params.lmSkip)
          })

        transducerLM
      }
      }


    transducerLM
  }

  def trainSkipEfficientCombinatoric(transducerLM: TransducerLM, partitionFunc: (Array[String]) => Array[Array[String]], filename: => String, params: Params, start: Int): TransducerLM = {

    Source.fromFile(filename).getLines().to(LazyList).map(line => line.trim)
      .filter(line => line.nonEmpty && line.length > params.lmMinSentenceLength && line.length < params.lmMaxSentenceLength)
      .zipWithIndex.filter(pair => pair._2 >= start)
      .take(params.lmMaxSentence).map(_._1)
      .foreach { line => {

        line.split("[\\s\\p{Punct}]+")
          .map(_.toLowerCase(locale))
          .filter(item => item.matches("\\p{L}+"))
          .filter(item => item.length < params.lmTokenLength)
          .sliding(params.lmWindowLength, 1)
          .toArray
          .foreach(sequence => transducerLM.countEfficientCombinatoric(sequence, partitionFunc, params.lmSample, params.lmWindowLength, params.lmSkip))

        transducerLM
      }
      }


    transducerLM
  }

  def prune(transducerLM: TransducerLM): TransducerLM = {
    prune(transducerLM)
  }

  def loadLM(filename: String): TransducerLM = {
    if (!new File(filename).exists()) {
      new TransducerLM(new Transducer())
    }
    else {
      println("Loading transducer from " + filename)
      val input = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filename)))
      val obj = new TransducerLM(new Transducer())
      obj.load(input)
      input.close()
      obj
    }
  }

  def loadLM(filename: => String, transducer: Transducer): TransducerLM = {
    if (!new File(filename).exists()) new TransducerLM(transducer)
    else {
      println("Loading transducer lm from " + filename)
      val input = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filename)))
      val obj = new TransducerLM(transducer)
      obj.load(input)
      input.close()
      obj
    }
  }

  def saveLM(filename: String, tranducerLM: TransducerLM): Unit = {
    println("Saving transducer lm")
    val output = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(filename)))
    tranducerLM.save(output)
    output.close()
  }

}

