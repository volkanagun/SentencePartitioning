package evaluation

import experiments.{LMDataset, Params}
import transducer.AbstractLM
import utils.Tokenizer

import java.io.{File, PrintWriter}
import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.control.NonFatal

class Ablation(tokenizer: Tokenizer = Ablation.defaultTokenizer()) {

  private val resultFolder = "resources/results/reviewer1"
  private val skipGramModel = "skip"
  private val evaluationTasks = Array("pos", "ner", "sentiment", "analogy")
  private val lmMethods = Array("frequent-ngram", "lm-lemma", "lm-rank", "lm-skip", "lm-syllable", "lm-subword")
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
                            status: String)

  def experiments(taskName: String, methodName: String): Unit = {
    val task = normalizeTask(taskName)
    val method = normalizeMethod(methodName)
    val outputDir = new File(resultFolder)
    outputDir.mkdirs()

    val rows = ablationParams(method).zipWithIndex.map { case (params, index) =>
      params.embeddingModel = skipGramModel
      params.adapterName = method
      runVariant(params, task, method, s"v${index + 1}")
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
    val parallelCombinations = combinations.par
    parallelCombinations.tasksupport = new ForkJoinTaskSupport(
      new ForkJoinPool(math.max(1, math.min(parallelEvaluations, combinations.length))))

    val rows = parallelCombinations.map { case (task, method) =>
      val start = System.currentTimeMillis()
      val status =
        try {
          new Ablation().experiments(task, method)
          "completed"
        }
        catch {
          case NonFatal(error) =>
            "failed: " + Option(error.getMessage).getOrElse(error.getClass.getSimpleName)
        }

      CombinationRow(task, method, status, System.currentTimeMillis() - start)
    }.toArray.sortBy(row => (row.task, row.method))

    writeCombinationCsv(new File(outputDir, "comment1-all-evaluations.csv"), rows)
    writeCombinationSummary(new File(outputDir, "comment1-all-evaluations.md"), rows)
  }

  private def runVariant(params: Params, task: String, method: String, variantId: String): RunRow = {
    val corpusTask = if (task == "analogy") "intrinsic" else task
    val corpusFilename = params.corpusFilename(corpusTask)
    val resultFilename = expectedResultFilename(params, corpusTask)

    val status =
      try {
        val lm = params.model(params, method)
        val trainedLM = lm.initialize().loadTrain()
        ensureCorpus(trainedLM, corpusTask)
        evaluate(params, task, trainedLM, corpusFilename, resultFilename)
      }
      catch {
        case NonFatal(error) =>
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
      status = status)
  }

  private def evaluate(params: Params,
                       task: String,
                       lm: AbstractLM,
                       corpusFilename: String,
                       resultFilename: String): String = {
    if (new File(resultFilename).exists()) {
      return "found"
    }
    if (!new File(corpusFilename).exists()) {
      return "missing-corpus"
    }

    val corpusTask = if (task == "analogy") "intrinsic" else task
    val mainEvaluation =
      if (corpusTask == "intrinsic") {
        new IntrinsicEvaluation(resultFilename)
          .attachEvaluations("resources/evaluation/analogy/sentence-tr.json")
          .compile()
      }
      else {
        val evaluation = new IntrinsicEvaluation(resultFilename)
        evaluation.functions :+= extrinsicFunction(params, corpusTask, lm)
        evaluation
      }

    if (corpusTask == "intrinsic") {
      mainEvaluation.filter(Array("SEMEVAL"))
    }

    val words = mainEvaluation.universe()
    val modelling = modellingPath(params, corpusTask)
    params.modelName(skipGramModel + "-" + modelling.hashCode.toString)
    val embeddingModel = params.createModel(params.embeddingModel, tokenizer, lm).train(corpusFilename)
    mainEvaluation.setDictionary(words, embeddingModel)
    mainEvaluation.evaluateReport(embeddingModel, params)
    "completed"
  }

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
    if (!corpusFile.exists() || params.forceTrain) {
      new LMDataset().construct(lm, task)
    }
  }

  private def expectedResultFilename(params: Params, task: String): String = {
    params.resultFilename(modellingPath(params, task))
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

    if (method == "lm-rank") {
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
    else if (method == "lm-lemma") {
      (base ++ Array(5, 7, 9).map(stemLength => {
        val params = tunedParams(method, 3, 3)
        params.lmStemLength = stemLength
        params
      })).distinctBy(_.lmID())
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

  private def tunedParams(method: String, window: Int, topSplit: Int): Params = {
    val params = Params(method, window)
    params.adapterName = method
    params.embeddingModel = skipGramModel
    params.epocs = 5
    params.batchSize = 128
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
      case other => throw new IllegalArgumentException("Unsupported evaluation strategy: " + other)
    }
  }

  private def normalizeMethod(methodName: String): String = {
    methodName.trim.toLowerCase match {
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
      writer.println("Morphology is excluded from this reviewer evaluation.")
      writer.println(s"Evaluation strategy: `$task`")
      writer.println(s"LM method: `$method`")
      writer.println("Embedding model: `SkipGramModel`")
      writer.println(s"Ablation variants: ${rows.length}")
      writer.println()
      writer.println("Evaluations use `ExtrinsicNER`, `ExtrinsicPOS`, `ExtrinsicSentiment`, or `IntrinsicEvaluation` depending on the selected strategy.")
      writer.println("The CSV file beside this summary records all tuned LM parameters and the result XML path for each variant.")
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
      writer.println("The no-argument `comment1()` entry point runs every evaluation strategy against every LM method in parallel.")
      writer.println(s"Parallel jobs: $parallelEvaluations")
      writer.println(s"Evaluation strategies: ${evaluationTasks.mkString(", ")}")
      writer.println(s"LM methods: ${lmMethods.mkString(", ")}")
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
      writer.println("Call `new Reviewer1().comment1()` to run every evaluation strategy and LM method in parallel.")
      writer.println()
      writer.println("Call `new Reviewer1().comment1(task, method)` for a targeted run with task `POS`, `NER`, `Sentiment`, or `Analogy` and method `FrequentLM`, `LemmaLM`, `RankLM`, `SkipLM`, `SyllableLM`, or `LMSubword`.")
      writer.println()
      writer.println("The runner trains or loads the selected LM, constructs the task corpus when needed, trains SkipGram on that corpus, and writes evaluation XML plus the reviewer CSV/summary under `resources/results`.")
      writer.println()
      writer.println("For `RankLM`, ablation includes Algorithm 4 likelihood/prior damping weights and length penalty formulations.")
    } finally {
      writer.close()
    }
  }

  private def csv(value: String): String = {
    val escaped = value.replace("\"", "\"\"")
    "\"" + escaped + "\""
  }
}

object Ablation {
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
