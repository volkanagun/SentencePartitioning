package evaluation

import experiments.Params
import transducer.AbstractLM
import utils.Tokenizer

/** Ten-fold NER evaluation over the union of the original training and testing documents. */
class CrossNEREvaluation(params: Params, tokenizer: Tokenizer, lm: AbstractLM)
  extends ExtrinsicNER(params, tokenizer, lm) {

  private lazy val combinedDocuments =
    CrossValidationSupport.readDocuments(getTraining()) ++
      CrossValidationSupport.readDocuments(getTesing())

  override def labels(): Array[String] = {
    if (categories == null) {
      println("Finding cross-validation category labels")
      categories = "NONE" +: combinedDocuments
        .flatMap(_.linesIterator)
        .filter(_.trim.nonEmpty)
        .map(_.trim.toLowerCase(locale).split("[\\t\\s]+").last)
        .toSet
        .toArray
        .sorted
        .filterNot(_ == "NONE")
    }
    categories
  }

  override def universe(): Set[String] = combinedDocuments
    .flatMap(_.linesIterator)
    .filter(_.trim.nonEmpty)
    .map(_.trim.split("[\\t\\s]+").head)
    .toSet

  override def evaluate(): EvalScore = CrossValidationSupport.evaluate(
    params,
    getClassifier(),
    combinedDocuments,
    System.lineSeparator() + System.lineSeparator(),
    () => load(),
    filename => iterator(filename),
    () => labels().length,
    detail => reportProgress(detail))
}
