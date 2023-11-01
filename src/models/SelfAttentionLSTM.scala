package models

import evaluation.{EvalScore, InstrinsicEvaluationReport, IntrinsicFunction}
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingSequenceLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sampling.experiments.SampleParams
import utils.Params

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.io.Source

class SelfAttentionLSTM(params: SampleParams) extends EmbeddingModel(params){

  def onehot(indices: Array[Int]): INDArray = {
    val ndarray = Nd4j.zeros(1, params.dictionarySize, params.modelWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })
    ndarray
  }


  def index(indices: Array[Int]): INDArray = {
    val ndarray = Nd4j.zeros(1,  params.modelWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._2)), pair._1.toFloat)
    })
    ndarray
  }


  def onehot(indices: Array[Int], categorySize:Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize, params.modelWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })
    ndarray
  }
 def onehot(indice: Int, categorySize:Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize)
    ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice)), 1f)
    ndarray
  }

  def vectorize(sentence: String): Array[INDArray] = {
    tokenize(sentence).sliding(params.modelWindowLength, params.modelWindowLength)
      .map(ngrams => onehot(ngrams.map(update(_))))
      .toArray
  }
  def vectorizeIndex(sentence: String): Array[INDArray] = {
    tokenize(sentence).sliding(params.modelWindowLength, params.modelWindowLength)
      .map(ngrams => index(ngrams.map(update(_))))
      .toArray
  }

  def vectorizeOneHotLast(sentence: String): Array[INDArray] = {
    tokenize(sentence).sliding(params.modelWindowLength, params.modelWindowLength)
      .map(ngrams => onehot(update(ngrams.last), params.dictionarySize))
      .toArray
  }

  def maskInput(sentenceVector: Array[INDArray]): Array[INDArray] = {
    sentenceVector.map(_ => {
      val mask = Nd4j.ones(1, params.modelWindowLength)
      mask.put(Array(NDArrayIndex.point(params.modelWindowLength - 1)), 0f)
      mask
    })
  }

  def maskOutput(sentenceVector: Array[INDArray]): Array[INDArray] = {
    sentenceVector.map(_ => {
      val mask = Nd4j.zeros(1, 1)
      mask.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0)), 1f)
      mask
    })
  }

  def iterator(filename: String): MultiDataSetIterator = {

    new MultiDataSetIterator {

      var lines = Source.fromFile(filename).getLines()
      override def next(i: Int): MultiDataSet = {
        var cnt = 0
        var trainStack = Array[INDArray]()
        var trainOutputStack = Array[INDArray]()
        var maskInputStack = Array[INDArray]()
        var maskOutputStack = Array[INDArray]()


        while(cnt < params.batchSize && hasNext){
          val sentence = lines.next()
          val sentenceVector = vectorizeIndex(sentence)
          val lastWordVector = vectorizeOneHotLast(sentence)

          trainStack = trainStack ++ sentenceVector
          trainOutputStack = trainOutputStack ++ lastWordVector
          maskInputStack = maskInputStack ++ maskInput(sentenceVector)
          maskOutputStack = maskOutputStack ++ maskOutput(sentenceVector)
          cnt += 1
        }

        val maskingInput = Nd4j.vstack(maskInputStack: _*)
        val maskingOutput = Nd4j.vstack(maskOutputStack: _*)
        val trainVector = Nd4j.vstack(trainStack: _*)
        val trainOutputVector = Nd4j.vstack(trainOutputStack: _*)
        new org.nd4j.linalg.dataset.MultiDataSet(trainVector, trainOutputVector, maskingInput, maskingOutput)
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = false

      override def reset(): Unit = {
        lines = Source.fromFile(filename).getLines()
      }

      override def hasNext: Boolean = lines.hasNext

      override def next(): MultiDataSet = next(0)
    }

  }


  override def train(filename: String): EmbeddingModel = {

    var i = 0
    val net = model()
    val statsStorage = new InMemoryStatsStorage()
    val uiServer = UIServer.getInstance()
    uiServer.attach(statsStorage)

    net.addListeners(new StatsListener(statsStorage, 1))

    val multiDataSetIterator = iterator(filename)

    var start = System.currentTimeMillis()
    var isTrained = false
    sampleCount = 0
    while (i < params.epocs) {

      println("Epoc : " + i)

      net.fit(multiDataSetIterator)
      multiDataSetIterator.reset()

      i = i + 1
      sampleCount += params.batchSize
      //System.gc()
    }
    val passedTime = System.currentTimeMillis() - start
    avgTime = passedTime / (sampleCount)
    println("Saving model...")
    net.save(new File(params.modelFilename()))
    computationGraph = net
    save(computationGraph)
    uiServer.stop()
    this

  }

  def save(net:ComputationGraph): EmbeddingModel = {
    val weights = net.getVertex("embedding").paramTable(false).get("W")
    //val weightsMatrix = weights.toFloatMatrix
    val objectOutput = new ObjectOutputStream(new FileOutputStream(params.dictionaryFilename()))
    objectOutput.writeInt(dictionaryIndex.size)
    dictionaryIndex.foreach{case(ngram, id) => {
      val embeddingVector = weights.get(NDArrayIndex.point(id)).toFloatVector
       objectOutput.writeObject(ngram)
       objectOutput.writeObject(embeddingVector)
      dictionary = dictionary.updated(ngram, embeddingVector)
    }}
    objectOutput.close()
    this
  }

  override def load(): SelfAttentionLSTM.this.type = {
    if(new File(params.dictionaryFilename()).exists()) {
      val objectInput = new ObjectInputStream(new FileInputStream(params.dictionaryFilename()))
      val size = objectInput.readInt()
      Range(0, size).foreach { index => {
        val ngram = objectInput.readObject().asInstanceOf[String]
        val vector = objectInput.readObject().asInstanceOf[Array[Float]]
        dictionary = dictionary.updated(ngram, vector)
        dictionaryIndex = dictionaryIndex.updated(ngram, index)
      }
      }
      objectInput.close()
    }
    this

  }

  def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .layer("embedding", new EmbeddingSequenceLayer.Builder()
        .inputLength(params.modelWindowLength)
        .nIn(params.dictionarySize)
        .nOut(params.embeddingLength).build(), "input")
      .layer("input-lstm", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.hiddenLength)
        .activation(Activation.TANH).build, "embedding")
      .layer("attention", new SelfAttentionLayer.Builder().nOut(params.hiddenLength).nHeads(params.nheads).projectInput(true).build(), "input-lstm")
      .layer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
      .layer("dense_base", new DenseLayer.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength).activation(Activation.RELU).build(), "pooling")
      .layer("dense", new DenseLayer.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength).activation(Activation.RELU).build(), "dense_base")
      .layer("output", new OutputLayer.Builder().nIn(params.hiddenLength).nOut(params.dictionarySize).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "dense")
      .setInputTypes(InputType.recurrent(params.dictionarySize))
      .build()

    new ComputationGraph(conf)
  }

  override def save(): EmbeddingModel = this

  override def evaluate(model: EmbeddingModel): EvalScore = EvalScore(0, 0)

  override def setDictionary(set: Set[String], model: EmbeddingModel): SelfAttentionLSTM.this.type = this

  override def setWords(set: Set[String]): SelfAttentionLSTM.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): SelfAttentionLSTM.this.type = this

  override def count(): Int = 0

  override def universe(): Set[String] = Set()

  override def getClassifier(): String = "self-attention"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
