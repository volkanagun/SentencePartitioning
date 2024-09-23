package utils

import experiments.Params

import java.io._
import java.util.Locale
import java.util.concurrent.ForkJoinPool
import java.util.regex.Pattern
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport
import scala.io.Source
import scala.util.Random
import scala.util.control.Breaks

/**
 * @Author Dr. Hayri Volkan Agun
 * @Date 15.03.2022 16:07
 * @Project BigLanguage
 * @Version 1.0
 */

@SerialVersionUID(1000L)
class Tokenizer(val modelFilename: String = "/resources/binary/dictionary", windowSize: Int = 3) extends Serializable {

  val txtFilename = new File("").getAbsoluteFile().getAbsolutePath + modelFilename + "-" + windowSize + ".bin"
  val zipFilename = new File("").getAbsoluteFile().getAbsolutePath + modelFilename + "-" + windowSize + ".zip"

  val regexWord = "([abcçdefgğhıijklmnoöprsştuüvyzwqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX\\p{L}]+)"
  val regexLongNum1 = "(\\d+[\\/\\-\\.\\\\]\\d+([\\/\\-\\.\\\\]?))+"
  val regexLongNum2 = "(\\d+[\\/\\-\\.\\\\]\\s\\d+\\s([\\/\\-\\.\\\\]?))+"
  val regexNum = "(\\d+)"
  val regexParanthesisOpen = "([\\(\\[\\{\"\\<\\`])"
  val regexParanthesisClose = "([\\}\\]\\)\"\\>\\`])"
  val regexSTOP = "([\\?\\.\\:\\!])"
  val regexSEP = "((\\_|\\-|\\&\\&|~|¨|\\'|\\|\\|\\,\\;))"
  val regexSYM = "([\\^\\%\\$\\#\\€\\Â\\*\\£\\=\\@])"
  val regexSPACE = "(\\s+)"

  val regexArray = Array(regexWord, regexLongNum1, regexLongNum2, regexNum, regexParanthesisOpen,
    regexParanthesisClose, regexSTOP, regexSEP, regexSYM)
  val patternArray = regexArray.map(Pattern.compile(_, Pattern.UNICODE_CHARACTER_CLASS))

  var frequency = Map[String, Double]()
  var frequencyBin = Map[String, Double]()
  val locale = new Locale("tr")
  var countSum = 0L


  def maskSymbols(sentence: String): String = {
    var masked = regexLongNum1.r.replaceAllIn(sentence, " NUM ")
    masked = regexLongNum2.r.replaceAllIn(masked, " NUM ")
    masked = regexParanthesisOpen.r.replaceAllIn(masked, " OPEN ")
    masked = regexParanthesisClose.r.replaceAllIn(masked, " CLOSE ")
    masked = regexNum.r.replaceAllIn(masked, " NUM ")
    masked = regexSEP.r.replaceAllIn(masked, " - ")
    masked
  }

  def freqConstruct(filename: String, dictionaryFilename:String): Tokenizer = {

    Source.fromFile(filename).getLines().foreach(sentence => {
      val tokens = standardTokenizer(sentence)
      val sentences = Array(tokens.mkString(" "))
      sentences.foreach(sentence => {
        val splited = sentence.split("\\s+")
        Range(1, 3).toArray.flatMap(s => splited.sliding(s).map(items => items.mkString(" ")).toArray)
          .foreach(item => {
            frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + 1d)
            countSum = countSum + 1
          })
      })
    })

    frequency = frequency.filter { case (item, count) => count > 2 }

    save(dictionaryFilename)
  }

  def freqConstructBySentence(sentence: String, cutoff: Int = 2): Tokenizer = {
    val tokens = standardTokenizer(sentence)
    val sentences = Array(tokens.mkString(" "))

    sentences.foreach(sentence => {
      val splited = sentence.split("\\s+")
      Range(1, 3).toArray.flatMap(s => splited.sliding(s).map(items => items.mkString(" ")).toArray)
        .foreach(item => {
          frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + 1d)
          countSum = countSum + 1
        })
    })

    frequency = frequency.filter { case (item, count) => count > cutoff }

    this
  }

  def freqDictionaryConstruct(filename: String, dictionaryFilename:String): Tokenizer = {
    val dictionaryItems = Source.fromFile(filename).getLines()
      .map(line => line.split("\\s?\\:\\s?")).map(items => items(0).trim
        .toLowerCase(locale)).map(item => item.replaceAll("(mak|mek)$", ""))
      .map(item => item.replaceAll("(.*?)(/\\s(.*?))((\\s(.*?))+)$", "$1$4"))
      .map(item => item.replaceAll("(.*?)/(\\s?)\\p{L}+$", "$1"))
      .map(item => item.replaceAll("[\\`\\’]", "'"))
      .map(item => item.replaceAll("â", "a"))
      .map(item => item.trim)
      .flatMap(item => Array(
        item.replaceAll("[\\(\\)]", ""),
        item.replaceAll("[\\(.*?\\)]", "")
      )).toArray.distinct

    dictionaryItems.map(phrase => {
      standardTokenizer(phrase)
    }).filter(tokens => tokens.size < 6).foreach(tokens => {
      val phrase = tokens.mkString(" ")
      val sentences = ngramStemCombinations(tokens)
      sentences.foreach(csen => {
        val splited = csen.split("\\s+")
        Range(1, 3).toArray.flatMap(s => splited.sliding(s).toArray
            .map(items => items.mkString(" ")))
          .foreach(item => {
            frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + 1d)
            countSum = countSum + 1
          })
      })

      frequency = frequency.updated(phrase, frequency.getOrElse(phrase, Double.MaxValue / 4.0))
    })

    save(dictionaryFilename)
  }

  def freqStemConstruct(filename: String, windowSize: Int = windowSize, maxSize: Int = 1000000): Tokenizer = {

    val rnd = new Random()
    val main = Source.fromFile(filename).getLines().filter(_.length < 80)
    val filtered = Source.fromFile(filename).getLines().filter(_.length < 80)
    val size = filtered.size
    val mxSize = math.min(size, maxSize)
    val rndSet = Range(0, mxSize)
      .map(_ => rnd.nextInt(size))
      .toArray

    main.zipWithIndex.filter(pair => rndSet.contains(pair._2))
      .take(mxSize)
      .map(_._1)
      .foreach(sentence => {
        println("Processing: " + sentence.trim)
        val newSentence = " " + sentence + " "
        freqConstruct(newSentence, windowSize)
      })

    this
  }

  def freqStemConstructByIndex(index: Int, filename: String, windowSize: Int = windowSize, maxSize: Int = 1000000): Tokenizer = {

    val main = Source.fromFile(filename).getLines().filter(_.length < 80).zipWithIndex.filter(pair => pair._2 >= index)
      .take(maxSize).map(_._1)

    main.foreach(sentence => {
      val newSentence = " " + sentence + " "
      freqConstruct(newSentence, windowSize)
    })

    this
  }

  def build(cutoff: Int = 5): this.type = {

    frequency = frequency.filter { case (item, count) => count >= cutoff }
    binCountUpdate()

    frequency = Map[String, Double]()
    countSum = 0
    this
  }

  def binCountUpdate(count: Int = 100000): this.type = {

    val values = frequency.map(_._2)
    if (!values.isEmpty) {
      val max = values.max
      val min = values.min
      val step = (max - min) / count
      val newBin = frequency.map { case (item, value) => {
        (item, value / step + 1d)
      }
      }

      newBin.foreach { case (item, value) => frequencyBin = frequencyBin.updated(item, frequencyBin.getOrElse(item, 0d) + value) }
    }

    this
  }

  def save(filename:String): Tokenizer = {
    println("Saving tokenizer....")
    val outputStream = new ObjectOutputStream(new FileOutputStream(filename))
    val array1 = frequency.toArray
    outputStream.writeInt(array1.size)

    for (i <- 0 until array1.size) {
      val item = array1(i)
      outputStream.writeObject(item)
    }

    val array2 = frequencyBin.toArray
    outputStream.writeInt(array2.size)

    for (i <- 0 until array2.size) {
      val item = array2(i)
      outputStream.writeObject(item)
    }

    outputStream.writeLong(countSum)
    outputStream.close()
    this
  }

  def saveZip(filename:String): Tokenizer = {

    val baos = new FileOutputStream(filename);
    val gzipOut = new GZIPOutputStream(baos);
    val objectOut = new ObjectOutputStream(gzipOut);
    objectOut.writeObject(frequency);
    objectOut.writeObject(frequencyBin);
    objectOut.writeObject(countSum);
    objectOut.close();
    this
  }


  def load(dictionaryFilename:String): Tokenizer = {



    if (new File(dictionaryFilename).exists()) {
      println("Loading tokenizer from "+dictionaryFilename)
      val inputStream = new ObjectInputStream(new FileInputStream(dictionaryFilename))
      val size1 = inputStream.readInt()
      var array1 = Array[(String, Double)]()
      println("Reading frequency: " + size1)

      for (i <- 0 until size1) {
        println("Reading " + i + "/" + size1)
        val item = inputStream.readObject().asInstanceOf[(String, Double)]
        array1 = array1 :+ item
      }

      frequency = array1.toMap

      val size2 = inputStream.readInt()
      var array2 = Array[(String, Double)]()
      println("Reading frequency bin: " + size2)

      for (i <- 0 until size2) {
        println("Reading " + i + "/" + size2)
        val item = inputStream.readObject().asInstanceOf[(String, Double)]
        array2 = array2 :+ item
      }

      this.frequencyBin = array2.toMap
      this.countSum = inputStream.readLong()

      inputStream.close()
      this

    }
    else {
      this
    }
  }

  def exists(zipFilename:String): Boolean = {
    new File(zipFilename).exists()
  }

  def loadZip(zipFilename:String): Tokenizer = {
    println("Loading ZIP from "+zipFilename)
    if(exists(zipFilename)) {
      val baos = new FileInputStream(zipFilename);
      val gzipOut = new GZIPInputStream(baos);
      val objectOut = new ObjectInputStream(gzipOut);
      frequency = objectOut.readObject().asInstanceOf[Map[String, Double]];
      frequencyBin = objectOut.readObject().asInstanceOf[Map[String, Double]];
      countSum = objectOut.readObject().asInstanceOf[Long]
      objectOut.close();
    }
    this
  }

  def merge(freqWordTokenizer: Tokenizer): Tokenizer = {
    freqWordTokenizer.frequency.foreach { case (item, value) => {
      frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + value)
    }
    }

    countSum = countSum + freqWordTokenizer.countSum
    this
  }


  def freqStemTokenizer(sentence: String): Array[Array[String]] = {

    standardTokenizer(sentence.toLowerCase(new Locale("tr")))
      .sliding(windowSize, windowSize).toArray
      .map(subtokens => ngramFilter(subtokens))
      .flatMap(combinations => {
        combinations.map(sequence => {
          val tokens = sequence.split("[\\#\\s]+").map(item => item.trim)
            .filter(_.nonEmpty)
          tokens
        })
      })
  }

  def characterTokenizer(sentence: String): Array[Array[String]] = {

    standardTokenizer(sentence.toLowerCase(new Locale("tr")))
      .sliding(windowSize, windowSize).toArray
      .flatMap(tokens => {
        tokens.map(token => {
          val characters = token.toCharArray.map(_.toString)
            .filter(_.nonEmpty) :+ "#"
          characters
        })
      })
  }

  def freqNGramTokenizer(sentence: String): Array[Array[String]] = {

    standardTokenizer(sentence.toLowerCase(new Locale("tr")))
      .sliding(windowSize, 1).toArray
      .map(subtokens => ngramStemCombinations(subtokens))
      .map(combinations => {


        val finalResult = combinations.map(sentence => {
          var result = Array[(String, Double)]()
          var tokens = sentence.split("\\s+")
            .flatMap(token => {
              val array = token.split("#")
              val farray = (
                if (array.length == 2) {
                  Array(array.head + "#", array.last)
                }
                else {
                  Array(token + "#")
                })

              farray
            })

          var i = 0
          val scanLength = 4
          while (i < tokens.length) {
            val max = Math.min(tokens.length, i + scanLength)
            var j = i + 1

            Breaks.breakable {
              while (j < max) {

                val slice = tokens.slice(i, j).mkString(" ")

                if (!frequency.contains(slice)) {
                  Breaks.break()
                }

                j = j + 1
              }
            }

            val index = Math.max(i + 1, j - 1)
            val slice = tokens.slice(i, index).mkString(" ")
            result = result :+ (slice, Math.log(1 + frequencyBin.getOrElse(slice, 0d)))
            i = index;
          }
          result
        })


        val all = finalResult.map(toks => (toks, toks.map(_._2).sum / (toks.length * 0.0)))
          .sortBy(_._2)

        all.last._1
          .map(_._1)

      })
  }

  def freqConstruct(sentence: String, window: Int = 10, ngrams: Int = 4): this.type = {

    standardTokenizer(sentence.toLowerCase(new Locale("tr")))
      .sliding(window, 1).toArray
      .map(subtokens => ngramStemCombinations(subtokens))
      .flatMap(combinations => combinations.map(token => token.split("[\\s\\#]")))
      .flatMap(sequence => sequence.sliding(ngrams))
      .foreach(combinations => {
        Range(1, ngrams).foreach(len => {
          combinations.sliding(len, 1).foreach(item => {
            val str = item.mkString(" ")
            frequency = frequency.updated(str, frequency.getOrElse(str, 0d) + 1d)
            countSum = countSum + 1
          })
        })
      })

    this
  }

  //separate everything
  def standardTokenizer(sentence: String, pattern: Pattern, start: Int): Option[(String, Int, Int)] = {

    val matcher = pattern.matcher(sentence)
    if (matcher.find(start)) {
      val (s, e) = (matcher.start(), matcher.end())
      val group = sentence.substring(s, Math.min(sentence.length, Math.max(e, s + 1))).trim
      if (!group.isEmpty) {
        return Some((group, s, e))
      }
    }

    None
  }

  def standardReplacer(sentence: String): String = {
    var masked = maskSymbols(sentence)
    patternArray.foreach(pattern => {
      masked = pattern.pattern().r.replaceAllIn(masked, " $1 ")
    })
    masked.replaceAll(regexSPACE, " ").trim
  }

  def wordTokenizer(sentence: String, start: Int): Option[(String, Int, Int)] = {
    val res = patternArray.flatMap(p => standardTokenizer(sentence, p, start)).toArray
      .sortBy(tuple => tuple._3)
    if (res.isEmpty) None
    else Some(res.head)
  }


  def standardTokenizer(sentence: String): Array[String] = {

    val presentence = " " + sentence + " "
    var start = 0
    val masked = standardReplacer(presentence)
    var found = wordTokenizer(masked, start)
    var array = Array[String]()
    while (found.isDefined) {
      val (group, start, end) = found.get
      array = array :+ group
      found = wordTokenizer(masked, end)
    }

    array
  }

  protected def combinatoric(input: Array[Array[String]], result: Array[Array[String]], i: Int = 0): Array[Array[String]] = {

    if (i >= input.length) result
    else {

      var crr = i;
      var array = Array[Array[String]]()

      for (k <- 0 until input(crr).length) {
        for (i <- 0 until result.length) {
          val current = result(i) :+ input(crr)(k)
          array = array :+ current
        }
      }

      combinatoric(input, array, crr + 1)
    }
  }

  def ngramCombinations(sentence: String, stemLength: Int = 5): Array[String] = {
    //find the root
    val arrays = standardTokenizer(sentence).map(token => token.sliding(stemLength, 1).toArray)
    combinatoric(arrays, Array(Array[String]())).map(tokens => tokens.mkString(" "))
  }


  def ngramStemPartition(token: String): Array[String] = {
    val removes = Array(0, 1, 2, 3, 4, 5, 9).reverse

    removes.map(cut => {
      val max = Math.min(Math.max(token.length - cut, 3), token.length)
      token.substring(0, max) + "#" + token.substring(max)
    }).distinct

  }


  def ngramStemCombinations(sentence: Array[String]): Array[String] = {
    //find the root
    val arrays = sentence.map(token => {
      ngramStemPartition(token).flatMap(token => token.split("[\\#\\s]+"))
        .map(_.trim)
        .filter(_.nonEmpty)
    })

    combinatoric(arrays, Array(Array[String]())).map(tokens => tokens.mkString(" "))
  }

  def ngramFilter(sentence: String): Array[String] = {
    ngramFilter(standardTokenizer(sentence))
  }

  def ngramFilter(sentence: Array[String], top: Int = 1): Array[String] = {

    val result = sentence.flatMap(token => {
      val candidates = ngramStemPartition(token)
        .map(_.trim)
        .filter(_.nonEmpty)

      val sorted = candidates.sortBy(item => {
        val subngram = item.split("[\\#\\s]+").head
        val frequencySum = subngram.length * frequencyBin.getOrElse(subngram, 0d)
        frequencySum
      }).reverse.head.split("[\\#\\s]")

      sorted
    }).mkString(" ")

    Array(result)
  }

  def ngramTokenFilter(token: String): Array[String] = {
    val stemSlice = Range(1, token.length).toArray
    val suffixSlice = Range(3, 7)

    stemSlice.flatMap(max=>{
      val stem = token.substring(0, max)
      val rest = token.substring(max)
      val array = suffixSlice.flatMap(slice=> {
        rest.sliding(slice, 1).toArray[String]
      }).toArray
      val result = stem +: array
      result
    })
  }

}

object Tokenizer {

  val sentenceFilename = "resources/text/sentences/sentences-tr.txt";
  val dictionaryFilename = "resources/text/dictionary/dictionary.txt";
  val dictionaryZipFilename = "resources/text/dictionary/dictionary.zip";
  val locale = new Locale("tr")
  val lmWindowSize = 2


  def freqStemConstruct(): Unit = {

    val maxSize = 1000
    val epocs = 5
    val cutOff = 3
    val mainFreqDictionary = new Tokenizer(windowSize = lmWindowSize)
      .load(dictionaryFilename)

    for (i <- 0 until epocs) {
      val array = Range(0, 12).toArray.par
      array.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(12))
      array.map(_ => {
        new Tokenizer(windowSize = lmWindowSize)
          .freqStemConstruct(sentenceFilename, lmWindowSize, maxSize)
      }).toArray.foldRight[Tokenizer](mainFreqDictionary) {
        case (freq, main) => main.merge(freq)
      }.build(cutOff).saveZip(dictionaryZipFilename)
    }
  }

  def freqStemConstruct(sentenceFilename: String, params: Params, zipFilename:String): Unit = {

    println("Training tokenizer with n-gram filter")
    val maxSize = params.lmMaxSentence
    val epocs = params.lmEpocs
    val cutOff = 3
    val mainFreqDictionary = new Tokenizer(windowSize = lmWindowSize)
      .load(dictionaryFilename)
    Range(0, epocs).sliding(params.lmThreads, params.lmThreads)
      .toArray
      .foreach(sequence => {
        sequence.toArray.par.map(index => {
          new Tokenizer(windowSize = lmWindowSize)
            .freqStemConstructByIndex(index, sentenceFilename, lmWindowSize, maxSize)
        }).toArray.foldRight[Tokenizer](mainFreqDictionary) {
          case (freq, main) => main.merge(freq)
        }.build(cutOff).saveZip(zipFilename)
      })


  }

  def tokenize(sentence: String): Array[String] = {
    val tokenizer = new Tokenizer().load(dictionaryFilename)
    tokenizer.standardTokenizer(sentence)
  }

  def tokenizeNgram(sentence: String): Array[Array[String]] = {
    val tokenizer = new Tokenizer().load(dictionaryFilename)
    tokenizer.freqStemTokenizer(sentence)
  }

  def saveBinary(): Unit = {
    new Tokenizer().load(dictionaryFilename).build().save(dictionaryFilename)
  }

  def loadBinary(): Tokenizer = {
    new Tokenizer().load(dictionaryFilename)
  }

  def test(): Unit = {

    val sen1 = " iki kutuplu bir ulus olduk ."
    val sen2 = " mülteci olmak zorunda kalmayalım  ."
    val sen3 = " neden insanları yanımızda görmüyoruz ? "
    val sen4 = " karın keyfini çıkartanlar arasında ayrıca siyahi öğrenciler de vardı . "
    val sen5 = " belki aşırı iyimserim, ama niçin olmasın ? "
    val sen6 = " su zengini bir ülke miyiz biz değiliz . "

    val tokenizer = new Tokenizer(windowSize = 1).load(dictionaryFilename).build()
    val array = Array(sen1, sen2, sen3, sen4, sen5, sen6)


    array.foreach(sentence => {
      println("Frequency N-gram:\n" + tokenizer
        .ngramFilter(tokenize(sentence)).mkString(" "))
    })

  }

  def zip(): Unit = {
    val tokenizer = new Tokenizer(windowSize = lmWindowSize).load(dictionaryFilename)
    tokenizer.saveZip(dictionaryZipFilename)
  }

  def main(args: Array[String]): Unit = {


    //test()


  }
}
