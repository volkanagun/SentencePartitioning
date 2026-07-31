package evaluation

import experiments.Params
import models.StorchExtrinsicTrainer
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator

import java.io.{File, PrintWriter}
import java.nio.file.Files
import scala.util.Random

/** Shared deterministic fold construction for the extrinsic cross-validation evaluators. */
private[evaluation] object CrossValidationSupport {

  def evaluate(params: Params,
               classifier: String,
               records: Vector[String],
               recordSeparator: String,
               loadEmbeddings: () => Unit,
               iterator: String => MultiDataSetIterator,
               labelCount: () => Int,
               reportProgress: String => Unit): EvalScore = {
    val folds = params.evalCrossValidationFolds
    require(folds >= 2, s"Cross-validation requires at least two folds: $folds")
    require(records.size >= folds,
      s"Cross-validation requires at least one record per fold: records=${records.size}, folds=$folds")

    val shuffled = new Random(params.evalCrossValidationSeed).shuffle(records)
    val indexed = shuffled.zipWithIndex
    val temporaryDirectory = Files.createTempDirectory(s"sentence-partitioning-$classifier-cv-").toFile
    val trainingFile = new File(temporaryDirectory, "train.txt")
    val testingFile = new File(temporaryDirectory, "test.txt")

    loadEmbeddings()
    try {
      val metrics = (0 until folds).map { fold =>
        val testing = indexed.collect { case (record, index) if index % folds == fold => record }
        val training = indexed.collect { case (record, index) if index % folds != fold => record }
        writeRecords(trainingFile, training, recordSeparator)
        writeRecords(testingFile, testing, recordSeparator)

        val foldProgress =
          s"$folds-fold cross-validation | task: $classifier | fold: ${fold + 1}/$folds | " +
            s"train: ${training.size} | test: ${testing.size} | seed: ${params.evalCrossValidationSeed}"
        System.err.println(foldProgress)
        reportProgress(foldProgress)

        StorchExtrinsicTrainer.trainAndEvaluate(
          classifier,
          iterator(trainingFile.getAbsolutePath),
          iterator(testingFile.getAbsolutePath),
          params.evalEpocs,
          params.lrate,
          params.embeddingLength,
          params.hiddenLength,
          labelCount(),
          params.crossValidationModelEvaluationFilename(fold),
          params.storchBatch)
      }

      val accuracy = metrics.map(_.accuracy).sum / folds
      val precision = metrics.map(_.precision).sum / folds
      val recall = metrics.map(_.recall).sum / folds
      val f1 = metrics.map(_.f1).sum / folds
      System.err.printf(
        "%d-fold cross-validation mean | task: %s | precision: %.6f | recall: %.6f | macro-F1: %.6f | accuracy: %.6f%n",
        Int.box(folds), classifier, Double.box(precision), Double.box(recall), Double.box(f1), Double.box(accuracy))
      reportProgress(s"$folds-fold cross-validation completed for $classifier")
      EvalScore(accuracy, f1, precision, recall)
    }
    finally {
      Files.deleteIfExists(trainingFile.toPath)
      Files.deleteIfExists(testingFile.toPath)
      Files.deleteIfExists(temporaryDirectory.toPath)
    }
  }

  def readLines(filename: String): Vector[String] = {
    val source = scala.io.Source.fromFile(filename, "UTF-8")
    try source.getLines().filter(_.trim.nonEmpty).toVector
    finally source.close()
  }

  def readDocuments(filename: String): Vector[String] = {
    val source = scala.io.Source.fromFile(filename, "UTF-8")
    try source.mkString.split("(?:\\r?\\n){2,}").map(_.trim).filter(_.nonEmpty).toVector
    finally source.close()
  }

  private def writeRecords(file: File, records: Seq[String], separator: String): Unit = {
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.write(records.mkString(separator))
      writer.write(separator)
    }
    finally writer.close()
  }
}
