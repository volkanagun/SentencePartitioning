package evaluation

import experiments.{LMDataset, Params}
import models.SkipGramModel
import transducer.AbstractLM
import utils.Tokenizer

import java.io.{File, PrintWriter}
import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.control.NonFatal

class Ablation(tokenizer: Tokenizer = Ablation.defaultTokenizer()) {

  private val resultFolder = "resources/results/reviewer1"
  private val skipGramModel = "skip"
  private val evaluationTasks = Array("pos", "ner", "sentiment", "analogy", "morphology")
  private val lmMethods = Array("lm-word", "frequent-ngram", "lm-lemma", "lm-rank", "lm-skip", "lm-syllable", "lm-subword")
  private val parallelEvaluations = 4
  private case class CombinationRow(task: String,
                                    method: String,
                                    status: String,
                                    elapsedMillis: Long)

  private case class RunRow(variantId: String,
                            task: String,
                            method: String,
                            lmWindowLength: Int,
                            lmSlideLength: Int,
                            lmTopSplit: Int,
                            lmSkip: Int,
                            lmStemLength: Int,
                            lmPrune: Int,
                            lmLikelihoodWeight: Double,
                            lmPriorWeight: Double,
                            lmLengthPenalty: String,
                            corpusFilename: String,
                            resultFilename: String,
                            accuracy: Double,
                            f1: Double,
                            status: String)

  def experiments(taskName: String, methodName: String): Unit = {
    val task = normalizeTask(taskName)
    val method = normalizeMethod(methodName)
    val total = ablationParams(method).length
    val progress = new Ablation.ProgressBar(s"$task/$method", total)
    experiments(task, method, Some(progress))
    progress.finish()
  }

  private def experiments(task: String, method: String, progress: Option[Ablation.ProgressBar]): Unit = {
    val outputDir = new File(resultFolder)
    outputDir.mkdirs()

    val rows = ablationParams(method).zipWithIndex.map { case (params, index) =>
      params.embeddingModel = skipGramModel
      params.adapterName = method
      val variantId = s"v${index + 1}"
      val reportStage = (detail: String) =>
        progress.foreach(_.update(s"$task/$method/$variantId: $detail"))
      val row = runVariant(params, task, method, variantId, reportStage)
      progress.foreach(_.tick(s"$task/$method/${row.variantId}: ${row.status}"))
      row
    }

    val basename = s"comment1-${task}-${method}"
    writeCsv(new File(outputDir, basename + ".csv"), rows)
    writeSummary(new File(outputDir, basename + ".md"), task, method, rows)
  }

  def experiments(): Unit = {
    val outputDir = new File(resultFolder)
    outputDir.mkdirs()
    writeDesignMatrix(new File(outputDir, "comment1-design-matrix.md"))

    val combinations = evaluationTasks.flatMap(task => lmMethods.map(method => task -> method))
    val total = combinations.map { case (_, method) => ablationParams(method).length }.sum
    val progress = new Ablation.ProgressBar("all ablation variants", total)
    val parallelCombinations = combinations.par
    parallelCombinations.tasksupport = new ForkJoinTaskSupport(
      new ForkJoinPool(math.max(1, math.min(parallelEvaluations, combinations.length))))

    val rows = parallelCombinations.map { case (task, method) =>
      val start = System.currentTimeMillis()
      val status =
        try {
          new Ablation().experiments(task, method, Some(progress))
          "completed"
        }
        catch {
          case NonFatal(error) =>
            "failed: " + Option(error.getMessage).getOrElse(error.getClass.getSimpleName)
        }

      CombinationRow(task, method, status, System.currentTimeMillis() - start)
    }.toArray.sortBy(row => (row.task, row.method))

    progress.finish()
    writeCombinationCsv(new File(outputDir, "comment1-all-evaluations.csv"), rows)
    writeCombinationSummary(new File(outputDir, "comment1-all-evaluations.md"), rows)
  }

  private def runVariant(params: Params,
                         task: String,
                         method: String,
                         variantId: String,
                         reportStage: String => Unit): RunRow = {
    val corpusTask = if (task == "analogy") "intrinsic" else task
    val corpusFilename = if (task == "morphology") "resources/evaluation/morphology" else params.corpusFilename(corpusTask)
    val resultFilename = if (task == "morphology") morphologyResultFilename(method, variantId, params) else expectedResultFilename(params, corpusTask)
    var accuracy = Double.NaN
    var f1 = Double.NaN

    val status =
      try {
        reportStage("checking result artifacts")
        val resultFile = new File(resultFilename)
        if (resultFile.exists()) {
          if (task == "morphology") {
            readMorphologyResult(resultFile).foreach { case (cachedAccuracy, cachedF1) =>
              accuracy = cachedAccuracy
              f1 = cachedF1
            }
          }
          reportStage("result found; skipped")
          "found"
        }
        else {
          reportStage("checking transducer artifact")
          val lm = params.model(params, method)
          if (task == "morphology") {
            reportStage(if (lm.exists()) "loading transducer" else "building transducer")
            val trainedLM = if (lm.exists()) lm else lm.initialize().loadTrain()
            reportStage("evaluating morphology")
            val score = new ExtrinsicMorphology(params, tokenizer, trainedLM).evaluate()
            accuracy = score.tp
            f1 = score.similarity
            writeMorphologyResult(resultFile, accuracy, f1)
            "completed"
          }
          else {
            setEmbeddingModelName(params, corpusTask)
            val hasEmbedding = embeddingArtifactExists(params)
            val hasCorpus = new File(corpusFilename).exists()
            reportStage(
              if (hasEmbedding) "embedding artifact found; corpus and transducer construction skipped"
              else if (hasCorpus) "corpus artifact found; corpus and transducer construction skipped"
              else if (lm.exists()) "transducer artifact found; loading it"
              else "building transducer")

            // Only build the transducer and corpus when neither downstream artifact exists.
            val evaluationLM =
              if (hasEmbedding || hasCorpus || lm.exists()) lm
              else lm.initialize().loadTrain()
            if (!hasEmbedding && !hasCorpus) {
              reportStage("building evaluation corpus")
              ensureCorpus(evaluationLM, corpusTask)
              reportStage("evaluation corpus completed")
            }

            evaluate(params, task, evaluationLM, corpusFilename, resultFilename, hasEmbedding, reportStage)
          }
        }
      }
      catch {
        case NonFatal(error) =>
          reportStage("failed: " + Option(error.getMessage).getOrElse(error.getClass.getSimpleName))
          "failed: " + Option(error.getMessage).getOrElse(error.getClass.getSimpleName)
      }

    RunRow(
      variantId = variantId,
      task = task,
      method = method,
      lmWindowLength = params.lmWindowLength,
      lmSlideLength = params.lmSlideLength,
      lmTopSplit = params.lmTopSplit,
      lmSkip = params.lmSkip,
      lmStemLength = params.lmStemLength,
      lmPrune = params.lmPrune,
      lmLikelihoodWeight = params.lmLikelihoodWeight,
      lmPriorWeight = params.lmPriorWeight,
      lmLengthPenalty = params.lmLengthPenalty,
      corpusFilename = corpusFilename,
      resultFilename = resultFilename,
      accuracy = accuracy,
      f1 = f1,
      status = status)
  }

  private def evaluate(params: Params,
                       task: String,
                       lm: AbstractLM,
                       corpusFilename: String,
                       resultFilename: String,
                       embeddingAvailable: Boolean,
                       reportStage: String => Unit): String = {
    if (!embeddingAvailable && !new File(corpusFilename).exists()) {
      return "missing-corpus"
    }

    val corpusTask = if (task == "analogy") "intrinsic" else task
    val mainEvaluation =
      if (corpusTask == "intrinsic") {
        new IntrinsicEvaluation(resultFilename)
          .withProgressReporter(reportStage)
          .attachEvaluations("resources/evaluation/analogy/sentence-tr.json")
          .compile()
      }
      else {
        val evaluation = new IntrinsicEvaluation(resultFilename).withProgressReporter(reportStage)
        val function = extrinsicFunction(params, corpusTask, lm)
          .withProgressReporter(reportStage)
        evaluation.functions :+= function
        evaluation
      }

    if (corpusTask == "intrinsic") {
      mainEvaluation.filter(Array("SEMEVAL"))
    }

    reportStage("preparing evaluation queries")
    val words = mainEvaluation.universe()
    setEmbeddingModelName(params, corpusTask)
    val embeddingModel = params.createModel(params.embeddingModel, tokenizer, lm)
      .withProgressReporter(reportStage)
    reportStage(
      if (embeddingAvailable) "loading existing SkipGram artifact"
      else "training SkipGram embeddings")
    embeddingModel match {
      case skipGram: SkipGramModel if embeddingArtifactExists(params) => skipGram.loadExisting()
      case _ if new File(params.embeddingsFilename()).exists() => embeddingModel.load()
      case _ => embeddingModel.train(corpusFilename)
    }
    reportStage(s"embedding model ready; dictionary size=${embeddingModel.dictionary.size}")
    reportStage(s"evaluating embeddings for ${words.size} query words")
    mainEvaluation.setDictionary(words, embeddingModel)
    reportStage("running evaluation report")
    mainEvaluation.evaluateReport(embeddingModel, params)
    reportStage("evaluation report completed")
    "completed"
  }

  private def setEmbeddingModelName(params: Params, task: String): Unit = {
    val modelling = modellingPath(params, task)
    params.modelName(skipGramModel + "-" + modelling.hashCode.toString)
  }

  private def embeddingArtifactExists(params: Params): Boolean =
    new File(params.embeddingsFilename()).exists() || new File(params.modelFilename()).exists()

  private def extrinsicFunction(params: Params, task: String, lm: AbstractLM): ExtrinsicLSTM = {
    task match {
      case "ner" => new ExtrinsicNER(params, tokenizer, lm)
      case "pos" => new ExtrinsicPOS(params, tokenizer, lm)
      case "sentiment" => new ExtrinsicSentiment(params, tokenizer, lm)
      case _ => throw new IllegalArgumentException("Unsupported extrinsic task: " + task)
    }
  }

  private def ensureCorpus(lm: AbstractLM, task: String): Unit = {
    val params = lm.getParams
    val corpusFile = new File(params.corpusFilename(task))
    if (!corpusFile.exists()) {
      new LMDataset().construct(lm, task)
    }
  }

  private def expectedResultFilename(params: Params, task: String): String = {
    params.resultFilename(modellingPath(params, task))
  }

  private def morphologyResultFilename(method: String, variantId: String, params: Params): String = {
    s"$resultFolder/morphology/comment1-morphology-$method-$variantId-${params.lmID()}.csv"
  }

  private def writeMorphologyResult(file: File, accuracy: Double, f1: Double): Unit = {
    Option(file.getParentFile).foreach(_.mkdirs())
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.println("accuracy,f1")
      writer.println(Array(score(accuracy), score(f1)).map(csv).mkString(","))
    } finally {
      writer.close()
    }
  }

  private def readMorphologyResult(file: File): Option[(Double, Double)] = {
    val source = Source.fromFile(file, "UTF-8")
    try {
      source.getLines().drop(1).find(_.trim.nonEmpty).flatMap(line => {
        val cells = line.split(",", -1).map(_.trim.stripPrefix("\"").stripSuffix("\""))
        if (cells.length >= 2 && cells(0).nonEmpty && cells(1).nonEmpty) {
          Some(cells(0).toDouble -> cells(1).toDouble)
        }
        else {
          None
        }
      })
    } finally {
      source.close()
    }
  }

  private def modellingPath(params: Params, task: String): String = {
    skipGramModel + "/" + params.adapterName + "/" + task + "/" + params.embeddingModel + "/"
  }

  private def ablationParams(method: String): Array[Params] = {
    val windows = Array(2, 3, 4)
    val topSplits = Array(1, 3, 5)

    val base = windows.flatMap(window => {
      topSplits.map(topSplit => tunedParams(method, window, topSplit))
    })

    if (method == "lm-word"){
      val params = new Params()
      params.adapterName = method
      params.embeddingModel = skipGramModel
      params.epocs = 5
      params.batchSize = 128
      params.storchBatch = 256
      params.forceTrain = false
      params.lmForceTrain = false
      Array(params)
    }
    else if (method == "lm-rank" || method == "lm-lemma") {
      rankParams(method, base)
    }
    else if (method == "lm-subword") {
      (base ++ Array(5, 10, 20).map(sample => {
        val params = tunedParams(method, 3, 3)
        params.lmDoSample = true
        params.lmSample = sample
        params
      })).distinctBy(_.lmID())
    }
    else {
      base.distinctBy(_.lmID())
    }
  }

  private def rankParams(method: String, base: Array[Params]): Array[Params] = {
    val rankWeights = Array(0d, 0.05d, 0.15d, 0.30d, 0.50d)
    val penalties = Array("none", "inverse_parts", "inverse_sqrt_parts", "inverse_log_parts")
    val algorithm4 = rankWeights.flatMap(weight => {
      penalties.map(penalty => {
        val params = tunedParams(method, 3, 3)
        params.lmLikelihoodWeight = weight
        params.lmPriorWeight = 1d - weight
        params.lmLengthPenalty = penalty
        params
      })
    })

    (base ++ algorithm4).distinctBy(_.lmID())
  }

  private def ablationVariantCount(method: String): Int = {
    ablationParams(method).length
  }

  private def totalAblationVariantCount(): Int = {
    evaluationTasks.length * lmMethods.map(ablationVariantCount).sum
  }

  private def tunedParams(method: String, window: Int, topSplit: Int): Params = {
    val params = Params(method, window)
    params.adapterName = method
    params.embeddingModel = skipGramModel
    params.epocs = 5
    params.batchSize = 128
    params.storchBatch = 256
    params.forceTrain = false
    params.lmForceTrain = false
    params.lmWindowLength = window
    params.lmSlideLength = window
    params.lmSkip = window
    params.lmTopSplit = topSplit
    params
  }

  private def normalizeTask(taskName: String): String = {
    taskName.trim.toLowerCase match {
      case "pos" => "pos"
      case "ner" => "ner"
      case "sentiment" => "sentiment"
      case "analogy" => "analogy"
      case "intrinsic" => "analogy"
      case "morphology" => "morphology"
      case other => throw new IllegalArgumentException("Unsupported evaluation strategy: " + other)
    }
  }

  private def normalizeMethod(methodName: String): String = {
    methodName.trim.toLowerCase match {
      case "wordlm" | "lm-word" => "lm-word"
      case "frequentlm" | "frequent-ngram" => "frequent-ngram"
      case "lemmalm" | "lm-lemma" => "lm-lemma"
      case "ranklm" | "lm-rank" => "lm-rank"
      case "skiplm" | "lm-skip" => "lm-skip"
      case "syllablelm" | "lm-syllable" => "lm-syllable"
      case "lmsubword" | "lm-subword" | "subwordlm" => "lm-subword"
      case other => throw new IllegalArgumentException("Unsupported LM method: " + other)
    }
  }

  private def writeCsv(file: File, rows: Array[RunRow]): Unit = {
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.println(Array(
        "variant_id",
        "task",
        "method",
        "lm_window_length",
        "lm_slide_length",
        "lm_top_split",
        "lm_skip",
        "lm_stem_length",
        "lm_prune",
        "lm_likelihood_weight",
        "lm_prior_weight",
        "lm_length_penalty",
        "corpus_filename",
        "result_filename",
        "accuracy",
        "f1",
        "status").mkString(","))

      rows.foreach(row => {
        writer.println(Array(
          row.variantId,
          row.task,
          row.method,
          row.lmWindowLength.toString,
          row.lmSlideLength.toString,
          row.lmTopSplit.toString,
          row.lmSkip.toString,
          row.lmStemLength.toString,
          row.lmPrune.toString,
          row.lmLikelihoodWeight.toString,
          row.lmPriorWeight.toString,
          row.lmLengthPenalty,
          row.corpusFilename,
          row.resultFilename,
          score(row.accuracy),
          score(row.f1),
          row.status).map(_.toString).map(csv).mkString(","))
      })
    } finally {
      writer.close()
    }
  }

  private def writeSummary(file: File, task: String, method: String, rows: Array[RunRow]): Unit = {
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.println("# Reviewer 1 Comment 1")
      writer.println()
      writer.println("Morphology is evaluated as a direct LM partitioning task without CBOW or SkipGram embeddings.")
      writer.println(s"Evaluation strategy: `$task`")
      writer.println(s"LM method: `$method`")
      if (task == "morphology") {
        writer.println("Embedding model: none; this task directly evaluates the selected `AbstractLM` child partitioning.")
      }
      else {
        writer.println("Embedding model: `SkipGramModel`")
      }
      writer.println(s"Ablation variants: ${rows.length}")
      writer.println()
      writer.println("Evaluations use `ExtrinsicNER`, `ExtrinsicPOS`, `ExtrinsicSentiment`, `IntrinsicEvaluation`, or direct `ExtrinsicMorphology` scoring depending on the selected strategy.")
      if (task == "pos" || task == "ner" || task == "sentiment") {
        writer.println("POS, NER, and Sentiment use their original fixed training and testing datasets.")
      }
      if (task == "morphology") {
        writer.println("The CSV file beside this summary records all tuned LM parameters and direct accuracy/F1 scores for each variant.")
      }
      else {
        writer.println("The CSV file beside this summary records all tuned LM parameters and the result XML path for each variant.")
      }
      if (task == "morphology" && rows.exists(row => !row.accuracy.isNaN && !row.f1.isNaN)) {
        val bestAccuracy = rows.filterNot(_.accuracy.isNaN).maxBy(_.accuracy)
        val bestF1 = rows.filterNot(_.f1.isNaN).maxBy(_.f1)
        writer.println()
        writer.println(s"Best accuracy: `${bestAccuracy.variantId}` accuracy=${score(bestAccuracy.accuracy)}, f1=${score(bestAccuracy.f1)}.")
        writer.println(s"Best F1: `${bestF1.variantId}` accuracy=${score(bestF1.accuracy)}, f1=${score(bestF1.f1)}.")
      }
      writer.println()
      writer.println("Status counts:")
      rows.groupBy(_.status).toArray.sortBy(_._1).foreach { case (status, statusRows) =>
        writer.println(s"- `$status`: ${statusRows.length}")
      }
    } finally {
      writer.close()
    }
  }

  private def writeCombinationCsv(file: File, rows: Array[CombinationRow]): Unit = {
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.println("task,method,status,elapsed_millis,result_csv,result_summary")
      rows.foreach(row => {
        val basename = s"comment1-${row.task}-${row.method}"
        writer.println(Array(
          row.task,
          row.method,
          row.status,
          row.elapsedMillis.toString,
          resultFolder + "/" + basename + ".csv",
          resultFolder + "/" + basename + ".md").map(csv).mkString(","))
      })
    } finally {
      writer.close()
    }
  }

  private def writeCombinationSummary(file: File, rows: Array[CombinationRow]): Unit = {
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.println("# Reviewer 1 Comment 1 All Evaluations")
      writer.println()
      writer.println("The no-argument `experiments()` entry point runs every evaluation strategy against every LM method in parallel.")
      writer.println(s"Parallel jobs: $parallelEvaluations")
      writer.println(s"Evaluation strategies: ${evaluationTasks.mkString(", ")}")
      writer.println(s"LM methods: ${lmMethods.mkString(", ")}")
      writer.println(s"Total ablation parameter variants: ${totalAblationVariantCount()}")
      writer.println()
      writer.println("Status counts:")
      rows.groupBy(_.status).toArray.sortBy(_._1).foreach { case (status, statusRows) =>
        writer.println(s"- `$status`: ${statusRows.length}")
      }
    } finally {
      writer.close()
    }
  }

  private def writeDesignMatrix(file: File): Unit = {
    file.getParentFile.mkdirs()
    val writer = new PrintWriter(file, "UTF-8")
    try {
      writer.println("# Reviewer 1 Comment 1 Design Matrix")
      writer.println()
      writer.println("Call `new Ablation().experiments()` to run every evaluation strategy and LM method in parallel.")
      writer.println()
      writer.println("Call `new Ablation().experiments(task, method)` for a targeted run with task `POS`, `NER`, `Sentiment`, `Analogy`, or `Morphology` and method `FrequentLM`, `LemmaLM`, `RankLM`, `SkipLM`, `SyllableLM`, or `LMSubword`.")
      writer.println()
      writer.println("The runner trains or loads the selected LM, constructs the task corpus when needed, trains SkipGram on that corpus for POS/NER/Sentiment/Analogy, and writes evaluation XML plus the reviewer CSV/summary under `resources/results`.")
      writer.println("Morphology ablations do not train CBOW or SkipGram; they directly partition `resources/evaluation/morphology` sentences with the selected `AbstractLM` child and report accuracy/F1.")
      writer.println()
      writer.println("For `RankLM` and `LemmaLM`, ablation includes Algorithm 4 likelihood/prior damping weights and length penalty formulations.")
      writer.println()
      writer.println(s"Total ablation parameter variants across all tasks: ${totalAblationVariantCount()}")
      writer.println()
      writer.println("Variants per method:")
      lmMethods.foreach(method => writer.println(s"- `$method`: ${ablationVariantCount(method)} per task"))
    } finally {
      writer.close()
    }
  }

  private def csv(value: String): String = {
    val escaped = value.replace("\"", "\"\"")
    "\"" + escaped + "\""
  }

  private def score(value: Double): String = {
    if (value.isNaN) "" else f"$value%.6f"
  }

}

object Ablation {
  private val progressWidth = 40

  private class ProgressBar(label: String, total: Int) {
    private var completed = 0
    private val startedAt = System.currentTimeMillis()

    Console.err.println(s"Total ablation parameter variants for $label: $total")
    printProgress("starting")

    def tick(detail: String): Unit = synchronized {
      completed = math.min(total, completed + 1)
      printProgress(detail)
    }

    def update(detail: String): Unit = synchronized {
      printProgress(detail)
    }

    def finish(): Unit = synchronized {
      completed = total
      printProgress("done")
      Console.err.println()
    }

    private def printProgress(detail: String): Unit = {
      val done = if (total == 0) progressWidth else (completed.toDouble / total.toDouble * progressWidth).toInt
      val bar = "#" * done + "-" * (progressWidth - done)
      val percent = if (total == 0) 100d else completed.toDouble * 100d / total.toDouble
      val elapsedSeconds = (System.currentTimeMillis() - startedAt) / 1000
      Console.err.print(f"\r[$bar%s] $completed%4d/$total%-4d $percent%6.2f%% $elapsedSeconds%4ds $label%s - $detail%s")
      Console.err.flush()
    }
  }

  def defaultTokenizer(): Tokenizer = {
    val tokenizer = new Tokenizer(windowSize = 2)
    val tokenizerFile = new File("resources/dictionary/dictionary.zip")
    if (tokenizerFile.exists()) tokenizer.loadZip(tokenizerFile.getPath) else tokenizer
  }

  def experiments(taskName: String, methodName: String): Unit = {
    new Ablation().experiments(taskName, methodName)
  }

  def main(args: Array[String]): Unit = {
    if (args.length >= 2) {
      experiments(args(0), args(1))
    }
    else {
      new Ablation().experiments()
    }
  }
}
