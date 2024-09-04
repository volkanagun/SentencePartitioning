package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import experiments.Params
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator
import utils.Tokenizer

import java.io.{File, FileOutputStream}

class SkipGramModel(params:Params, tokenizer: Tokenizer) extends CBOWModel(params, tokenizer) {

  override def train(filename: String): EmbeddingModel = {
    val iter = new LineSentenceIterator(new File(filename))
    val factory = defaultTokenizer()
    val fname = params.modelFilename()

    if (!(new File(fname).exists())|| params.forceTrain) {
      println("SkipGram filename: " + fname)


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
        .elementsLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram")
        .allowParallelTokenization(true)
        .build()

      vectorModel.fit()
      WordVectorSerializer.writeWord2Vec(vectorModel, new FileOutputStream(fname))
      save()
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
