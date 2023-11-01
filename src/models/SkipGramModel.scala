package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import sampling.experiments.SampleParams
import utils.Params

import java.io.File

class SkipGramModel(params:SampleParams) extends CBOWModel(params) {

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
        .windowSize(params.modelWindowLength)
        .epochs(params.epocs)
        .batchSize(params.batchSize)
        .seed(42)
        .iterate(iter)
        .iterations(1)
        .tokenizerFactory(factory)
        .elementsLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram")
        .allowParallelTokenization(true)
        .build()

      vectorModel.fit()
    }

    this
  }

  override def save(): EmbeddingModel = this

  override def load(): SkipGramModel.this.type = this

  override def evaluate(model: EmbeddingModel): EvalScore = EvalScore(0d, 0d)

  override def setDictionary(set: Set[String], model: EmbeddingModel): SkipGramModel.this.type = this

  override def setWords(set: Set[String]): SkipGramModel.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): SkipGramModel.this.type = this

  override def count(): Int = 0

  override def universe(): Set[String] = Set()

  override def getClassifier(): String = "SkipGram"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
