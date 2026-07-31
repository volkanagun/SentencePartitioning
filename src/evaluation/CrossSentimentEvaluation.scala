package evaluation

import experiments.Params
import transducer.AbstractLM
import utils.Tokenizer

/** Ten-fold sentiment evaluation over the union of all original train/test examples. */
class CrossSentimentEvaluation(params: Params, tokenizer: Tokenizer, lm: AbstractLM)
  extends ExtrinsicSentiment(params, tokenizer, lm) {

  private lazy val combinedRecords =
    CrossValidationSupport.readLines(getTraining()).filter(_.contains("\t")) ++
      CrossValidationSupport.readLines(getTesing()).filter(_.contains("\t"))

  override def loadSamples(filename: String): Iterator[(String, String)] =
    CrossValidationSupport.readLines(filename).filter(_.contains("\t")).iterator.map { line =>
      val fields = line.toLowerCase(locale).split("\t", -1)
      (fields.dropRight(1).mkString(" "), fields.last)
    }

  override def labels(): Array[String] = {
    if (categories == null) {
      categories = combinedRecords
        .map(_.toLowerCase(locale).split("\t", -1).last)
        .toSet
        .toArray
        .sorted
      maxWindowSize = params.evalWindowLength
      println("Max window Size: " + maxWindowSize)
      println("Cross-validation category size: " + categories.length)
    }
    categories
  }

  override def universe(): Set[String] = combinedRecords.flatMap { line =>
    val fields = line.split("\t", -1)
    tokenizer.standardTokenizer(fields.dropRight(1).mkString(" ").toLowerCase(locale))
  }.toSet

  override def evaluate(): EvalScore = CrossValidationSupport.evaluate(
    params,
    getClassifier(),
    combinedRecords,
    System.lineSeparator(),
    () => load(),
    filename => iterator(filename),
    () => labels().length,
    detail => reportProgress(detail))
}
