package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import experiments.Params
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import transducer.AbstractLM
import utils.Tokenizer

import java.io._
import scala.collection.JavaConverters._

class CBOWModel(params:Params, tokenizer: Tokenizer, lm:AbstractLM) extends EmbeddingModel(params, tokenizer, lm) {

  var vectorModel:Word2Vec = null
  def defaultTokenizer(): DefaultTokenizerFactory = {
   val tokenizerFactory = new DefaultTokenizerFactory(){
     override def create(toTokenize: String): org.deeplearning4j.text.tokenization.tokenizer.Tokenizer = {
       val ngrams = tokenize(toTokenize).mkString(" ")
       super.create(ngrams)
     }

     override def create(toTokenize: InputStream): org.deeplearning4j.text.tokenization.tokenizer.Tokenizer = {
       throw new UnsupportedOperationException("Stream tokenizer is unsupported")
     }
   }

    tokenizerFactory
  }
  override def train(filename: String): EmbeddingModel = {
    val iter = new LineSentenceIterator(new File(filename))
    val factory = defaultTokenizer()
    val fname = params.modelFilename()
    val embeddingFile = params.embeddingsFilename()
    if (!(new File(fname).exists()) || !(new File(embeddingFile).exists()) || params.forceTrain) {
      println("Training for CBOW filename: " + fname)

      vectorModel = new Word2Vec.Builder()
        .workers(48)
        .minWordFrequency(params.freqCutoff)
        .layerSize(params.embeddingLength)
        .windowSize(params.embeddingWindowLength)
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
      WordVectorSerializer.writeWord2Vec(vectorModel, new FileOutputStream(fname))
      save()
    }
    else{
      load()
    }

    this
  }

  override def save(): EmbeddingModel = {
    if(vectorModel!=null){
      println("Saving vector model...")
      val filename = params.embeddingsFilename()
      val table = vectorModel.getLookupTable()
      val vocabCache = table.getVocabCache()
      val vectors = table.vectors()
      val printer = new ObjectOutputStream(new FileOutputStream(filename))

      val array = vectors.asScala.toArray
      printer.writeInt(array.length)

      array.zipWithIndex.foreach {indPair => {
        val word = vocabCache.elementAtIndex(indPair._2)
        val wordStr = word.getWord
        val wordVector = indPair._1.toFloatVector

        printer.writeObject(wordStr)
        printer.writeObject(wordVector)

        update(wordStr, wordVector)

      }}

      printer.close()
    }

    this
  }

  override def load(): EmbeddingModel = {
    val filename = params.embeddingsFilename()
    if(new File(filename).exists()) {
      println("Loading embedding filename: "+ filename)
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

  def load(filename:String): EmbeddingModel = {
    if(new File(filename).exists()) {
      println("Loading embedding filename: "+ filename)
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
