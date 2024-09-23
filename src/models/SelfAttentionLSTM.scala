package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import experiments.Params
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration
import org.nd4j.linalg.api.memory.enums.{LearningPolicy, ResetPolicy}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import smile.nlp.embedding.GloVe
import transducer.AbstractLM
import utils.Tokenizer

import java.io._
import scala.io.Source

class SelfAttentionLSTM(params: Params, tokenizer: Tokenizer, lm:AbstractLM) extends EmbeddingModel(params, tokenizer, lm) {

  def onehot(indices: Array[Int]): INDArray = {
    val ndarray = Nd4j.zeros(1, params.dictionarySize, params.embeddingWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })

    for (i <- indices.length until params.embeddingWindowLength) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }

    ndarray
  }


  def index(indices: Array[Int]): INDArray = {
    val ndarray = Nd4j.zeros(1, params.embeddingWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._2)), pair._1.toFloat)
    })

    for (i <- indices.length until params.embeddingWindowLength) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }


    ndarray
  }


  def onehot(indices: Array[Int], categorySize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize, params.embeddingWindowLength)
    indices.zipWithIndex.foreach(pair => {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(pair._1), NDArrayIndex.point(pair._2)), 1f)
    })

    for (i <- indices.length until params.embeddingWindowLength) {
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    }

    ndarray
  }

  def onehot(indice: Int, categorySize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize)
    ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice)), 1f)
    ndarray
  }

  def onehot(seqSize: Int, indice: Int, categorySize: Int): INDArray = {
    val ndarray = Nd4j.zeros(1, categorySize, seqSize)
    ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(indice), NDArrayIndex.point(seqSize - 1)), 1f)
    ndarray
  }

  def vectorize(array: Array[Array[Float]]): INDArray = {
    val sz1 = params.embeddingWindowLength
    val sz2 = params.embeddingLength
    val ndarray = Nd4j.zeros(1, sz2, sz1)
    array.zipWithIndex.foreach { case (sub, windex) => {
      val embedding = Nd4j.create(sub)
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(windex)), embedding)
    }
    }

    ndarray
  }
  def vectorize(array: Array[Array[Float]], size:Int): INDArray = {
    val sz1 = size
    val sz2 = params.embeddingLength
    val ndarray = Nd4j.zeros(1, sz2, sz1)
    array.zipWithIndex.foreach { case (sub, windex) => {
      val embedding = Nd4j.create(sub)
      ndarray.put(Array(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(windex)), embedding)
    }
    }

    ndarray
  }

  def vectorize(sentence: String): Array[INDArray] = {
    tokenize(sentence).sliding(params.embeddingWindowLength, params.embeddingWindowLength)
      .map(ngrams => onehot(ngrams.map(update(_))))
      .toArray
  }

  def vectorizeIndex(sentence: Array[String]): Array[INDArray] = {
    sentence.sliding(params.embeddingWindowLength, 1)
      .map(ngrams => index(ngrams.map(update(_))))
      .toArray
  }

  def vectorizeIndex(sentence: Array[String], winSize:Int, size:Int): Array[INDArray] = {
    sentence.sliding(winSize, 1)
      .map(ngrams => index(ngrams.take(winSize - 1).map(update(_, size))))
      .toArray
  }

  def vectorizeOneHotLast(sentence: Array[String]): Array[INDArray] = {
    sentence.sliding(params.embeddingWindowLength, 1)
      .map(ngrams => onehot(update(ngrams.last), params.dictionarySize))
      .toArray
  }

  def vectorizeOneHotLast(sentence: Array[String],winSize:Int, size:Int): Array[INDArray] = {
    sentence.sliding(winSize, 1)
      .map(ngrams => onehot(update(ngrams.last, size), size))
      .toArray
  }

  def maskInput(sentenceVector: Array[INDArray]): Array[INDArray] = {
    sentenceVector.map(_ => {
      val mask = Nd4j.ones(1, params.embeddingWindowLength)
      mask.put(Array(NDArrayIndex.point(params.embeddingWindowLength - 1)), 0f)
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

  def maskInput(maxSize: Int, originalSize:Int): INDArray = {
    val mask = Nd4j.zeros(1, maxSize)
    val beginIndex = maxSize - originalSize
    for(i<-beginIndex until maxSize) mask.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(i)), 1f)
    mask
  }

  def maskOutput(size: Int): INDArray = {
    val mask = Nd4j.zeros(1, size)
    mask.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(size - 1)), 1f)
    mask
  }
  def maskOutput(): INDArray = {
    val mask = Nd4j.zeros(1, 1)
    mask.put(Array(NDArrayIndex.point(0), NDArrayIndex.point(0)), 1f)
    mask
  }

  def getSize(filename:String):Int ={
    val set = Source.fromFile(filename).getLines().flatMap(line=> line.split("\\s+")).toSet
    set.size
  }



  def iterator(filename: String): MultiDataSetIterator = {
    val dictionarySize = getSize(filename)

    new MultiDataSetIterator {

      var max = 256
      var lines = Source.fromFile(filename).getLines()
      var senCount = 0
      var cacheInputArray = Array[INDArray]()
      var cacheOutputArray = Array[INDArray]()

      def getData():(INDArray, INDArray)={
        if(cacheInputArray.isEmpty){
          senCount+=1
          val sentence = lines.next()
          val frequentNgrams = sentence.split("\\s+")
          val sentenceVector = vectorizeIndex(frequentNgrams,params.embeddingWindowLength, dictionarySize)
          val lastWordVector = vectorizeOneHotLast(frequentNgrams, params.embeddingWindowLength,dictionarySize)
          val finalSentenceVector = sentenceVector
          val finalLastWordVector = lastWordVector
          cacheInputArray = cacheInputArray ++ finalSentenceVector
          cacheOutputArray = cacheOutputArray ++ finalLastWordVector
        }

        val inputArray = cacheInputArray.head
        val outputArray = cacheOutputArray.head

        cacheInputArray = cacheInputArray.drop(1)
        cacheOutputArray = cacheOutputArray.drop(1)

        (inputArray, outputArray)
      }

      override def next(i: Int): MultiDataSet = {
        var cnt = 0
        var trainStack = Array[INDArray]()
        var trainOutputStack = Array[INDArray]()

        println("Current sentence index: " + senCount)
        while (cnt < params.batchSize && hasNext) {

          val (finalSentenceVector, finalLastWordVector) = getData()
          trainStack = trainStack :+ finalSentenceVector
          trainOutputStack = trainOutputStack :+ finalLastWordVector

          cnt += 1

        }


        val trainVector = Nd4j.vstack(trainStack: _*)
        val trainOutputVector = Nd4j.vstack(trainOutputStack: _*)
        new org.nd4j.linalg.dataset.MultiDataSet(trainVector, trainOutputVector)
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = false

      override def reset(): Unit = {
        lines = Source.fromFile(filename).getLines()
      }

      override def hasNext: Boolean = lines.hasNext || cacheInputArray.nonEmpty

      override def next(): MultiDataSet = next(0)
    }

  }


  override def train(filename: String): EmbeddingModel = {

    var i = 0
    val fname = params.modelFilename()
    val modelFile = new File(fname)
    println("LSTM filename: " + fname)


    if (!(modelFile.exists()) || params.forceTrain) {


      val size = Source.fromFile(filename).getLines().size
      val dictionarySize = getSize(filename)

      load()

      computationGraph = model(dictionarySize)
      computationGraph.addListeners(new PerformanceListener(1, true))

      val multiDataSetIterator = iterator(filename)

      val start = System.currentTimeMillis()

      sampleCount = 0
      while (i < params.epocs) {

        println("Epoc : " + i)

        computationGraph.fit(multiDataSetIterator)
        multiDataSetIterator.reset()

        i = i + 1
        sampleCount += size
        //System.gc()
      }
      val passedTime = System.currentTimeMillis() - start
      avgTime = passedTime / (sampleCount)
      println("Saving model...")
      ModelSerializer.writeModel(computationGraph, modelFile, true)
      //uiServer.stop()
      System.gc()
      save()
    }
    else {
      computationGraph = ModelSerializer.restoreComputationGraph(modelFile)
    }
    this

  }

  def save(net: ComputationGraph): EmbeddingModel = {
    val weights = net.getLayer("embedding").paramTable(false).get("W")
    //val weightsMatrix = weights.toFloatMatrix
    val objectOutput = new ObjectOutputStream(new FileOutputStream(params.embeddingsFilename()))
    objectOutput.writeInt(dictionaryIndex.size)
    dictionaryIndex.foreach { case (ngram, id) => {
      val embeddingVector = weights.get(NDArrayIndex.point(id)).toFloatVector
      objectOutput.writeObject(ngram)
      objectOutput.writeObject(embeddingVector)
      dictionary = dictionary.updated(ngram, embeddingVector)
    }}
    objectOutput.close()
    this
  }

  override def load(): SelfAttentionLSTM.this.type = {
    val dictionaryFilename = params.embeddingsFilename()
    if (new File(dictionaryFilename).exists()) {
      println("Loading embedding filename: " + dictionaryFilename)
      val objectInput = new ObjectInputStream(new FileInputStream(params.embeddingsFilename()))
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



  def model(dictionarySize:Int): ComputationGraph = {


    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.PREFER_FASTEST)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .layer("embedding", new EmbeddingSequenceLayer.Builder()
        .inputLength(params.embeddingWindowLength)
        .nIn(dictionarySize)
        .nOut(params.embeddingLength).build(), "input")
      .layer("input-lstm", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.hiddenLength)
        .activation(Activation.TANH).build, "embedding")
      .layer("attention", new SelfAttentionLayer.Builder().nOut(params.hiddenLength).nHeads(params.nheads).projectInput(true).build(), "input-lstm")
      .layer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
      .layer("dense_base", new DenseLayer.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength).activation(Activation.RELU).build(), "pooling")
      .layer("dense", new DenseLayer.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength).activation(Activation.RELU).build(), "dense_base")
      .layer("output", new OutputLayer.Builder().nIn(params.hiddenLength).nOut(dictionarySize).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "dense")
      .setInputTypes(InputType.recurrent(dictionarySize))
      .build()

    conf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED)

    new ComputationGraph(conf)
  }


  def model(): ComputationGraph = {

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.PREFER_FASTEST)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .layer("embedding", new EmbeddingSequenceLayer.Builder()
        .inputLength(params.embeddingWindowLength)
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

    conf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED)

    new ComputationGraph(conf)
  }


  override def save(): EmbeddingModel = {
    save(computationGraph)
    this
  }

  override def evaluate(model: EmbeddingModel): EvalScore = EvalScore(0, 0)

  override def setDictionary(set: Set[String], model: EmbeddingModel): SelfAttentionLSTM.this.type = this

  override def setWords(set: Set[String]): SelfAttentionLSTM.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): SelfAttentionLSTM.this.type = this

  override def count(): Int = 0

  override def universe(): Set[String] = Set()

  override def getClassifier(): String = "self-attention"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: Params): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
