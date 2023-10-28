package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import utils.{Params, Tokenizer}

import collection.JavaConverters._
import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream, PrintWriter}

class CBOWModel(params:Params) extends EmbeddingModel(params) {

  var vectorModel:Word2Vec = null
  def defaultTokenizer(): DefaultTokenizerFactory = {
    lazy val tokenPreProcess: TokenPreProcess = new TokenPreProcess {
      val tokenizer = new Tokenizer()
      override def preProcess(s: String): String = {
        tokenizer.ngramFilter(s).mkString(" ")
      }

    }

    val factory = new DefaultTokenizerFactory()
    factory.setTokenPreProcessor(tokenPreProcess)
    factory
  }
  override def train(filename: String): EmbeddingModel = {
    val iter = new LineSentenceIterator(new File(filename))
    val factory = defaultTokenizer()
    val fname = params.modelFilename()

    if (!(new File(fname).exists())) {
      println("CBOW filename: " + fname)
      vectorModel = new Word2Vec.Builder()
        .workers(params.nthreads)
        .minWordFrequency(params.freqCutoff)
        .layerSize(params.embeddingLength)
        .windowSize(params.windowLength)
        .epochs(params.epocs)
        .batchSize(params.batchSize)
        .seed(42)
        .iterate(iter)
        .iterations(1)
        .tokenizerFactory(factory)
        .elementsLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW")
        .allowParallelTokenization(true)
        .build()

      vectorModel.fit()
    }

    this
  }

  override def save(): EmbeddingModel = {
    if(vectorModel!=null){
      val filename = params.dictionaryFilename()
      val table = vectorModel.getLookupTable()
      val vocabCache = table.getVocabCache()
      val vectors = table.vectors()
      val printer = new ObjectOutputStream(new FileOutputStream(filename))
      var i = 0;
      val array = vectors.asScala.toArray
      printer.writeInt(array.length)

      array.foreach {ind=>{
        val word = vocabCache.elementAtIndex(i)
        val wordStr = word.getWord
        val wordVector = ind.toFloatVector

        printer.writeObject(wordStr)
        printer.writeObject(ind.toFloatVector)

        update(wordStr, wordVector)

      }}

      printer.close()
    }

    this
  }

  override def load(): EmbeddingModel = {
    val filename = params.dictionaryFilename()
    if(new File(filename).exists()) {
      val reader = new ObjectInputStream(new FileInputStream(filename))
      val size = reader.readInt()
      for (i <- 0 until size) {
        val wordStr = reader.readObject().asInstanceOf[String]
        val wordVector = reader.readObject().asInstanceOf[Array[Float]]
        update(wordStr, wordVector)
      }
      reader.close()
    }
    this
  }

  override def evaluate(model: EmbeddingModel): EvalScore = ???

  override def setDictionary(set: Set[String], model: EmbeddingModel): CBOWModel.this.type = this

  override def setWords(set: Set[String]): CBOWModel.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): CBOWModel.this.type = this

  override def count(): Int = 0

  override def universe(): Set[String] = Set()

  override def getClassifier(): String = "CBOW"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: Params): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
