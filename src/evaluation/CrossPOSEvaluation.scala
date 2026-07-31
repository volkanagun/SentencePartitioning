package evaluation

import experiments.Params
import transducer.AbstractLM
import utils.Tokenizer

import scala.util.Random

/** Ten-fold POS evaluation over the union of the original training and testing sentences. */
class CrossPOSEvaluation(params: Params, tokenizer: Tokenizer, lm: AbstractLM)
  extends ExtrinsicPOS(params, tokenizer, lm) {

  private lazy val combinedRecords =
    CrossValidationSupport.readLines(getTraining()) ++ CrossValidationSupport.readLines(getTesing())

  override def loadSamples(filename: String): Iterator[(String, String)] = {
    val records = CrossValidationSupport.readLines(filename)
    new Random(17).shuffle(records).iterator.map { line =>
      val input = line.split("\\t").head
      (input.toLowerCase(locale), input.toLowerCase(locale))
    }
  }

  override def labels(): Array[String] = {
    if (categories == null) {
      println("Finding cross-validation category labels")
      categories = "NONE" +: combinedRecords
        .flatMap(_.split("\\t").head.toLowerCase(locale).split("\\s+").map(_.split("/").last))
        .toSet
        .toArray
        .sorted
        .filterNot(_ == "NONE")
    }
    categories
  }

  override def universe(): Set[String] = combinedRecords
    .flatMap(_.split("\\t").head.split("\\s+").map(_.split("/").head))
    .map(_.trim)
    .filter(_.nonEmpty)
    .toSet

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
