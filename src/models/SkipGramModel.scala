package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import experiments.Params
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import transducer.AbstractLM
import utils.Tokenizer

import java.io.{File, FileOutputStream}

class SkipGramModel(params:Params, tokenizer: Tokenizer,  lm:AbstractLM) extends CBOWModel(params, tokenizer, lm) {

  /** Loads a persisted SkipGram artifact without entering the training path. */
  def loadExisting(): EmbeddingModel = {
    val embeddings = new File(params.embeddingsFilename())
    val modelFile = new File(params.modelFilename())
    if (embeddings.exists()) {
      reportProgress("loading existing SkipGram embeddings")
      load()
    }
    else if (modelFile.exists()) {
      reportProgress("loading SkipGram model " + modelFile.getPath)
      vectorModel = WordVectorSerializer.readWord2VecModel(modelFile)
      save()
    }
    else {
      throw new IllegalStateException("SkipGram model does not exist: " + modelFile.getPath)
    }
    this
  }

  override def train(filename: String): EmbeddingModel = {
    val fname = params.modelFilename()
    val embeddingFile = params.embeddingsFilename()
    val modelFile = new File(fname)
    val embeddings = new File(embeddingFile)

    if (embeddings.exists()) {
      loadExisting()
    }
    else if (modelFile.exists()) {
      loadExisting()
    }
    else {
      reportProgress("training SkipGram model " + fname)
      val iter = new LineSentenceIterator(new File(filename))
      val factory = defaultTokenizer()
      val windowLength = 5

      vectorModel = new Word2Vec.Builder()
        .workers(32)
        .minWordFrequency(3)
        .layerSize(params.embeddingLength)
        .windowSize(windowLength)
        .epochs(params.epocs)
        .batchSize(params.batchSize)
        .seed(42)
        .iterate(iter)
        .iterations(1)
        .tokenizerFactory(factory)
        .elementsLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram")
        .allowParallelTokenization(true)
        // DL4J 1.0.0-M2.1 cannot combine hierarchical softmax and negative
        // sampling when a vocabulary entry has no Huffman codes: SkipGram
        // passes an empty code array to Nd4j.create(), which rejects it.
        // Use the already configured negative-sampling objective by itself.
        .useHierarchicSoftmax(false)
        .sampling(0.3)
        .negativeSample(5)
        .build()

      vectorModel.fit()
      reportProgress("saving trained SkipGram model " + fname)
      WordVectorSerializer.writeWord2Vec(vectorModel, new FileOutputStream(fname))
      save()
      reportProgress("SkipGram training completed")
    }

    this
  }

  override def evaluate(model: EmbeddingModel): EvalScore = EvalScore(0d, 0d)

  override def setDictionary(set: Set[String], model: EmbeddingModel): SkipGramModel.this.type = this

  override def setWords(set: Set[String]): SkipGramModel.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): SkipGramModel.this.type = this

  override def count(): Int = 0

  override def getClassifier(): String = "SkipGram"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: Params): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
