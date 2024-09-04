package tagging.hmm

import experiments.Params
import transducer.{RegularFix, TransducerLM, TransducerOp}

import java.io.File
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source
import scala.util.Random

class SentenceHMM() extends SequenceHMM() {

  val endMarker = "END"
  val epsMarker = "EPS"
  val splitMaker = "#"

  val regularFix = new RegularFix()
  val parent = new File(".")
    .getAbsoluteFile()
    .getParentFile()
    .getAbsolutePath

  val modelFilename = s"${parent}/resources/binary/sentence-split.bin"
  val transducerFilename = s"${parent}/resources/binary/syllables.bin"
  val textFilename = s"${parent}/resources/text/sentences/sentences-tr.txt"
  val dictionaryTextFilename = "resources/dictionary/lexicon.txt"


  var transducerLM: TransducerLM = new TransducerLM(TransducerOp.fromSyllables())

  var topSplit = 1
  val epocs = 1

  val maxSentences = 25000
  val maxSentenceLength = 180

  val candidateEnding = Array(
    "[\\:\\.\\!\\?]+(\")$",
    "[du|dü|di|dı|tu|tü|ti|tı|ma|me|yor|yör|uz|iz|üz|öz|ız|dur|dir|m|n](\\s?)[\\:\\.\\!\\?]+$",
    "[lar|ler|lir|lır|tir|tır|tur|tür](\\s?)[\\:\\.\\!\\?]+$",
    "[\\)\\]\\}](\\s?)[\\:\\.\\!\\?]+$",
    "[\\:\\.\\!\\?]+(\\s+?)$"
  )

  def trainLM(): this.type = {

    println("Training LM ...")
    val topSplit = 3;
    val params = new Params()

    params.adapterName = "lm-syllable"
    params.lmEpocs = 1
    params.lmMaxSentence = 500000
    params.lmForceTrain = true
    params.lmCandidateCount = 3
    params.lmMaxSentenceLength = 170
    params.lmWindowLength = 10

    transducerLM = new TransducerLM(TransducerOp.fromSyllables())
    val infer: (String => Array[String]) = (input: String) => {
      transducerLM.transducer.multipleSplitSearch(input, topSplit)
        .flatMap(seq => seq.split(transducerLM.transducer.split)).map(_.trim)
        .filter(_.nonEmpty)
    }

    val seqTransducer = transducerLM.seqTransducer
    TransducerOp.fromDictionaryByInfer(infer, seqTransducer, dictionaryTextFilename, params)
    TransducerOp.fromTextByInfer(infer, seqTransducer, textFilename, params)
    TransducerOp.saveLM(transducerFilename, transducerLM)
    this
  }

  def load(): SentenceHMM = {

    if (new File(modelFilename).exists()) {
      load(modelFilename)
       transducerLM = TransducerOp.loadLM(transducerFilename)
    }
    else {
      trainLM()
      trainEpocs()
    }


    this
  }


  def retrieveLM(sentence: Array[String]): Array[(String, String)] = {
    sentence.flatMap(token => {
      val splits = transducerLM.infer(token, topSplit)
        .head
        .split(transducerLM.seqTransducer.marker)
      splits.map(sp => (token, sp))
    })
  }

  def retrieveTokenLM(sentence: Array[String]): Array[(String, String)] = {
    sentence.map(token => (transducerLM.infer(token, topSplit).head, epsMarker))
  }

  def candidates(sentence: Array[String]): Array[(String, String)] = {
    sentence.map(token => {
      if (candidateFilter(token)) {
        (token, endMarker)
      }
      else {
        (token, epsMarker)
      }
    }).map(token => (transducerLM.infer(token._1, topSplit).head, token._2))
  }

  def candidateFilter(token: String): Boolean = {
    candidateEnding.exists(regex => !regex.r.findAllMatchIn(token).isEmpty)
  }

  def candidateSplit(text:String):Array[String]={
    val fixed = regularFix.fixSentence(text)
    val tokens = regularFix.fixTokens(fixed)
    var sentence = ""
    var sentences = Array[String]()
    tokens.foreach(token=> {
      sentence += token + " "
      if(candidateFilter(token)){
        sentences :+= sentence.trim
        sentence = ""
      }
    })

    if(sentence.nonEmpty) sentences :+ sentence
    else sentences
  }

  var hmm = load()

  def exists(): Boolean = {
    new File(modelFilename).exists()
  }


  def combineEnd(array: Array[(String, String)]): Array[(String, String)] = {
    val length = array.length
    val heads = array.take(length - 1)
    heads :+ (array(length - 1)._1, endMarker)
  }

  def combineSplit(array: Array[(String, String)]): Array[(String, String)] = {
    array.flatMap { case (input, output) => {
      val split = input.split(splitMaker)
      if (output.equals(endMarker)) {
        split.take(split.length - 1).map(item => (item, epsMarker)) :+ (split.last, endMarker)
      }
      else {
        split.map(item => (item, output))
      }
    }
    }
  }

  def trainLarge(): SentenceHMM = {
    println("Training large....")
    //val ints = Range(0, maxSentences)
    //  .map(ii => Random.nextInt(maxLines))
    Source.fromFile(textFilename).getLines().zipWithIndex
      //.filter(pair => ints.contains(pair._2))
      .map(pair => pair._1)
      .filter(sentence=> sentence.length < maxSentenceLength)
      .take(maxSentences)
      .map(line => regularFix.fixSentence(line))
      .sliding(2, 1).zipWithIndex.toArray.par.map { case (Seq(first, second), index) => {
        println("Training sentence combination: " + index)
        val firstPatterns = combineSplit(combineEnd(retrieveTokenLM(first.split("\\s+"))))
        val sequence = firstPatterns ++ combineSplit(combineEnd(retrieveTokenLM(second.split("\\s+"))))
        val input = sequence.map(_._1)
        val output = sequence.map(_._2)
        (input, output)
      }
      }.toArray.zipWithIndex.foreach { case ((input, output), index) => {
        println("Processing index: " + index)
        train(input, output)
      }
      }


    println("Training finished.")

    this
  }

  def trainEpocs(): SentenceHMM = {

    for (i <- 0 until epocs) {
      println("Training sentence epocs: " + i)
      trainLarge()
      save(modelFilename)
    }

    normalize()
    this
  }

  /*

    def train(): SentenceHMM = {
      val ints = Range(0, maxLines)
        .map(ii => Random.nextInt(maxLines))

      Source.fromFile(textFilename).getLines().take(maxLines).zipWithIndex.filter(pair => ints.contains(pair._2))
        .map(pair => pair._1)
        .map(line => {
          regularFix.fixSentence(line)
        }).filter(line => candidateFilter(line)).zipWithIndex
        .foreach(pair => {

          println("Processing index: " + pair._2)
          val line = pair._1
          val tokens = line.split("\\s+")
          val sequence = candidates(tokens).flatMap { case (inp, out) => {
            val split = inp.split("#")
            if (!out.equals(endMarker)) {
              split.map(item => (item, out))
            }
            else {
              split.take(split.length - 1).map(item => (item, epsMarker)) :+ (split.last, endMarker)
            }
          }
          }

          val input = sequence.map(_._1)
          val output = sequence.map(_._2)

          if (!sequence.isEmpty) {
            train(input, output)
          }
        })

      this
    }
  */

  def train():SentenceHMM={
    trainEpocs()
    save(modelFilename)
    this
  }

  def loadTrain(): SentenceHMM = {

    if (exists()) {
      load(modelFilename)
    }

    trainEpocs()
    save(modelFilename)
    this
  }

  def find(tokens: Array[String]): Array[String] = {
    val retrieved = retrieveLM(tokens)
    val results = inferByTokens(tokens, retrieved)
    results
  }

  def split(tokens: Array[String]): Array[String] = {
    var sentence = ""
    var sentences = Array[String]()
    val marks = find(tokens)
    marks.foreach(token => {
      var ftoken = token
      ftoken = ftoken.replaceAll(endMarker, "")
      ftoken = ftoken.replaceAll(epsMarker, "")
      ftoken = ftoken.replaceAll(labelMarker, "")
      ftoken = ftoken.replaceAll(start, "")
      sentence = sentence + ftoken + " "
      if (token.endsWith(endMarker)) {
        sentences :+= sentence.trim
        sentence = ""
      }
    })

    sentences
  }

  def split(text: String): Array[String] = {
    val fixed = regularFix.fixSentence(text)
    val tokens = regularFix.fixTokens(fixed)
    val sentences = split(tokens)
    sentences
  }


}

object SentenceHMM {

  def test(): Unit = {

    val hmm = new SentenceHMM().loadTrain()
    val text = "i̇ki ton patlayıcı yüklü traktörle yapılan intihar saldırısında, 2 asker şehit oldu, 24 asker ise yaralandı. " +
      "açıklamada, karakola teröristlerce 2 ton bomba yüklü bir traktörle intihar saldırısı düzenlendiği, saldırı sonucunda 2 askerin şehit olduğu, 24 askerin de yaralandığı açıklandı." +
      "kazada sürücü ahmet yeltekin (28) ile yanında oturan hamza demiray (55) öldü, araçta bulunan alihan öztürkoğlu (34) ise yaralandı. " +
      "kaza, i̇negöl'de akşam saatlerinde meydana geldi'."

    val text2 = "adam bana çarptı."
    val maks1 = hmm.split(text2)

    maks1.foreach(array => {
      println(array.mkString(""))
    })

  }

  def main(args: Array[String]): Unit = {
    test()
  }
}