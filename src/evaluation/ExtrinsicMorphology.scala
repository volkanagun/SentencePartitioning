package evaluation

import experiments.Params
import models.EmbeddingModel
import transducer.AbstractLM
import utils.Tokenizer
import zemberek.morphology.TurkishMorphology
import zemberek.morphology.analysis.{SentenceWordAnalysis, SingleAnalysis}

import java.io.File
import java.util.Locale
import scala.io.Source
import scala.jdk.CollectionConverters._

class ExtrinsicMorphology(params: Params, tokenizer: Tokenizer, lm: AbstractLM) extends ExtrinsicFunction {

  val morphologyFolder = "resources/evaluation/morphology"
  val locale = new Locale("tr")

  private case class MorphUnits(lexemes: Set[String], suffixes: Set[String]) {
    def tagged: Set[String] = lexemes.map("LEX:" + _) ++ suffixes.map("SUF:" + _)
  }

  private case class Counts(tp: Double = 0d, fp: Double = 0d, fn: Double = 0d) {
    def +(other: Counts): Counts = Counts(tp + other.tp, fp + other.fp, fn + other.fn)

    def accuracy: Double = {
      val denominator = tp + fn
      if (denominator == 0d) 0d else tp / denominator
    }

    def f1: Double = {
      val denominator = 2d * tp + fp + fn
      if (denominator == 0d) 0d else (2d * tp) / denominator
    }
  }

  override def getClassifier(): String = "morphology"

  override def filter(group: Array[String]): Boolean = true

  override def train(filename: String): this.type = this

  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = this

  override def setWords(set: Set[String]): this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): this.type = this

  override def count(): Int = sentences().length

  override def universe(): Set[String] = {
    sentences().flatMap(sentence => tokenize(sentence)).toSet
  }

  def evaluate(): EvalScore = {
    val zemberek = TurkishMorphology.createWithDefaults()
    val counts = sentences().foldLeft(Counts()) { case (total, sentence) =>
      total + compare(sentence, zemberek)
    }

    EvalScore(counts.accuracy, counts.f1)
  }

  override def evaluate(model: EmbeddingModel): EvalScore = evaluate()

  override def evaluateReport(model: EmbeddingModel, embedParams: Params): InstrinsicEvaluationReport = {
    val report = new InstrinsicEvaluationReport().incrementTestPair()
    val classifier = getClassifier()
    val score = evaluate(model)

    report.incrementQueryCount(classifier, count().toDouble)
    report.incrementTruePositives(score.tp)
    report.incrementScoreMap(classifier, score.tp)
    report.incrementSimilarity(score.similarity)
    report.incrementSimilarityMap(classifier, score.similarity)
    report.printProgress(classifier)
    report
  }

  private def compare(sentence: String, zemberek: TurkishMorphology): Counts = {
    val tokens = tokenize(sentence)
    val gold = zemberekSentenceUnits(sentence, zemberek)
    val predicted = lmSentenceUnits(tokens)
    val truePositive = gold.intersect(predicted).size.toDouble
    val falsePositive = predicted.diff(gold).size.toDouble
    val falseNegative = gold.diff(predicted).size.toDouble
    Counts(truePositive, falsePositive, falseNegative)
  }

  private def zemberekSentenceUnits(sentence: String, zemberek: TurkishMorphology): Set[String] = {
    zemberek.analyzeAndDisambiguate(sentence)
      .getWordAnalyses
      .asScala
      .filter(wordAnalysis => containsLetter(wordAnalysis.getWordAnalysis.getInput))
      .zipWithIndex
      .flatMap { case (wordAnalysis, index) =>
        indexed(index, zemberekUnits(wordAnalysis))
      }
      .toSet
  }

  private def zemberekUnits(wordAnalysis: SentenceWordAnalysis): MorphUnits = {
    val bestAnalysis = wordAnalysis.getBestAnalysis
    if (bestAnalysis != null && !bestAnalysis.isUnknown) {
      zemberekUnits(bestAnalysis, wordAnalysis.getWordAnalysis.getInput)
    }
    else {
      MorphUnits(Set(normalize(wordAnalysis.getWordAnalysis.getInput)), Set())
    }
  }

  private def zemberekUnits(analysis: SingleAnalysis, token: String): MorphUnits = {
    val stem = normalize(Option(analysis.getStem).getOrElse(token))
    val surfacePieces = analysis.getMorphemeDataList.asScala
      .map(_.surface)
      .filter(surface => surface != null && surface.trim.nonEmpty)
      .map(normalize)
      .toArray

    val suffixes = {
      val fromMorphemes =
        if (surfacePieces.headOption.contains(stem)) surfacePieces.drop(1)
        else surfacePieces.filterNot(_ == stem)

      if (fromMorphemes.nonEmpty) fromMorphemes
      else Option(analysis.getEnding).map(normalize).filter(_.nonEmpty).toArray
    }

    MorphUnits(Set(stem).filter(_.nonEmpty), suffixes.toSet)
  }

  private def lmSentenceUnits(tokens: Array[String]): Set[String] = {
    if (tokens.isEmpty) {
      return Set()
    }

    val normalizedTokens = tokens.map(normalize)
    val segments = lm.splitSentence(tokens)
      .map(normalize)
      .filter(_.nonEmpty)

    alignSegments(normalizedTokens, segments)
      .zipWithIndex
      .flatMap { case (parts, index) =>
        val units =
          if (parts.isEmpty) MorphUnits(Set(normalizedTokens(index)), Set())
          else MorphUnits(Set(parts.head), parts.tail.toSet)
        indexed(index, units)
      }
      .toSet
  }

  private def alignSegments(tokens: Array[String], segments: Array[String]): Array[Array[String]] = {
    var segmentIndex = 0
    tokens.map(token => {
      var consumed = ""
      var parts = Array[String]()

      while (segmentIndex < segments.length && consumed.length < token.length) {
        val segment = segments(segmentIndex)
        parts :+= segment
        consumed += segment
        segmentIndex += 1
      }

      if (parts.isEmpty || consumed != token) Array(token)
      else parts
    })
  }

  private def indexed(index: Int, units: MorphUnits): Set[String] = {
    units.tagged.map(unit => s"$index:$unit")
  }

  private def containsLetter(item: String): Boolean = {
    item != null && item.exists(_.isLetter)
  }

  private def sentences(): Array[String] = {
    val folder = new File(morphologyFolder)
    if (!folder.exists() || !folder.isDirectory) {
      Array()
    }
    else {
      folder.listFiles()
        .filter(file => file.isFile)
        .sortBy(_.getName)
        .flatMap(file => {
          val source = Source.fromFile(file, "UTF-8")
          try {
            source.getLines().map(_.trim).filter(_.nonEmpty).toArray
          } finally {
            source.close()
          }
        })
    }
  }

  private def tokenize(sentence: String): Array[String] = {
    tokenizer.standardTokenizer(sentence.toLowerCase(locale))
      .map(_.trim)
      .filter(token => token.nonEmpty && containsLetter(token))
  }

  private def normalize(item: String): String = {
    item.toLowerCase(locale)
      .replaceAll("[\\#\\$\\s]+", "")
      .trim
  }
}
