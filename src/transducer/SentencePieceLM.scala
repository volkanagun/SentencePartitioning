package transducer

import experiments.Params
import org.bytedeco.sentencepiece.{SentencePieceProcessor, SentencePieceTrainer, Status, StringStringMap}

import java.io.File
import scala.collection.concurrent.TrieMap

/**
 * SentencePiece baseline backed by the official C++ unigram/BPE trainer and
 * processor through the JavaCPP preset.
 *
 * SentencePiece training is corpus based rather than incremental. Consequently
 * initialize/loadTrain train from AbstractLM.textFilename when the model
 * artifact is absent, while the incremental AbstractLM training hooks are
 * intentionally no-ops.
 */
class SentencePieceLM(override val params: Params) extends AbstractLM(params) {

  val modelPrefix: String =
    s"${parent}/resources/transducers/sentencepiece${params.lmID()}"
  val modelFilename: String = modelPrefix + ".model"
  val vocabularyFilename: String = modelPrefix + ".vocab"

  @transient private var processor: SentencePieceProcessor = _

  override def initialize(): this.type = loadTrain()

  override def isEmpty(): Boolean = !exists()

  override def copy(): AbstractLM = {
    val copied = new SentencePieceLM(params)
    if (copied.exists()) copied.load() else copied
  }

  override def save(): AbstractLM = this

  override def exists(): Boolean =
    new File(modelFilename).isFile && new File(vocabularyFilename).isFile

  override def load(transducer: Transducer): AbstractLM = load()

  override def load(): this.type = synchronized {
    if (!exists()) {
      throw new IllegalStateException(
        s"SentencePiece model does not exist: $modelFilename")
    }

    closeProcessor()
    val loaded = new SentencePieceProcessor()
    requireOk(loaded.Load(modelFilename), s"load SentencePiece model $modelFilename")
    processor = loaded
    this
  }

  override def loadTrain(): this.type = {
    SentencePieceLM.withModelLock(modelFilename) {
      if (!exists()) {
        trainModel()
      }
      load()
    }
  }

  override def trainSentence(sentence: String): AbstractLM = this

  override def train(sequence: Array[String]): AbstractLM = this

  override def train(sequence: String): AbstractLM = this

  override def trainDictionary(item: String): AbstractLM = this

  override def trainDictionary(item: Array[String]): AbstractLM = this

  override def splitSentence(sentence: Array[String]): Array[String] = {
    if (sentence.isEmpty) Array.empty
    else encode(sentence.mkString(" "))
  }

  override def splitToken(token: String): Array[String] = {
    val value = Option(token).map(_.trim).getOrElse("")
    if (value.isEmpty) Array.empty else encode(value)
  }

  override def normalize(): AbstractLM = this

  override def prune(): AbstractLM = this

  private def trainModel(): Unit = {
    validateParameters()
    val inputFile = new File(textFilename)
    if (!inputFile.isFile) {
      throw new IllegalArgumentException(
        s"SentencePiece training corpus does not exist: ${inputFile.getPath}")
    }

    Option(new File(modelFilename).getParentFile).foreach(_.mkdirs())
    val options = new StringStringMap()
    try {
      options.put("input", inputFile.getAbsolutePath)
      options.put("model_prefix", new File(modelPrefix).getAbsolutePath)
      options.put("model_type", params.sentencePieceModelType)
      options.put("vocab_size", params.sentencePieceVocabSize.toString)
      options.put(
        "character_coverage",
        params.sentencePieceCharacterCoverage.toString)
      options.put("hard_vocab_limit", "false")
      options.put("shuffle_input_sentence", "false")
      options.put("num_threads", math.max(1, params.nthreads).toString)
      options.put(
        "max_sentence_length",
        math.max(1, params.lmMaxSentenceLength).toString)
      options.put("bos_id", "-1")
      options.put("eos_id", "-1")
      options.put("pad_id", "-1")
      if (params.lmMaxSentence > 0) {
        options.put("input_sentence_size", params.lmMaxSentence.toString)
      }

      Console.err.println(
        s"Training SentencePiece ${params.sentencePieceModelType} model: " +
          s"input=${inputFile.getPath}, vocab=${params.sentencePieceVocabSize}, " +
          s"coverage=${params.sentencePieceCharacterCoverage}, " +
          s"output=$modelFilename")
      requireOk(
        SentencePieceTrainer.Train(options),
        s"train SentencePiece model $modelFilename")
    } finally {
      options.close()
    }

    if (!exists()) {
      throw new IllegalStateException(
        s"SentencePiece training completed without creating $modelFilename")
    }
  }

  private def encode(text: String): Array[String] = {
    val pieces = activeProcessor().EncodeAsPieces(text)
    try {
      val result = pieces.get().filter(_.nonEmpty)
      if (result.nonEmpty) result else Array(text)
    } finally {
      pieces.close()
    }
  }

  private def activeProcessor(): SentencePieceProcessor = {
    if (processor == null) {
      load()
    }
    processor
  }

  private def closeProcessor(): Unit = {
    if (processor != null) {
      processor.close()
      processor = null
    }
  }

  private def validateParameters(): Unit = {
    val modelTypes = Set("unigram", "bpe", "char", "word")
    if (!modelTypes.contains(params.sentencePieceModelType)) {
      throw new IllegalArgumentException(
        "Unsupported SentencePiece model type: " +
          params.sentencePieceModelType +
          s"; expected one of ${modelTypes.toArray.sorted.mkString(", ")}")
    }
    if (params.sentencePieceVocabSize < 4) {
      throw new IllegalArgumentException(
        "SentencePiece vocabulary size must be at least 4")
    }
    if (params.sentencePieceCharacterCoverage <= 0d ||
      params.sentencePieceCharacterCoverage > 1d) {
      throw new IllegalArgumentException(
        "SentencePiece character coverage must be in (0, 1]")
    }
    if (params.lmMaxSentence > 0 && params.lmMaxSentence <= 100) {
      throw new IllegalArgumentException(
        "SentencePiece input sentence size must be 0 or greater than 100")
    }
  }

  private def requireOk(status: Status, operation: String): Unit = {
    try {
      if (!status.ok()) {
        throw new IllegalStateException(
          s"Failed to $operation: ${status.ToString()}")
      }
    } finally {
      status.close()
    }
  }
}

object SentencePieceLM {
  private val modelLocks = TrieMap.empty[String, AnyRef]

  private[transducer] def withModelLock[T](filename: String)(action: => T): T = {
    modelLocks.getOrElseUpdate(filename, new Object()).synchronized(action)
  }
}
