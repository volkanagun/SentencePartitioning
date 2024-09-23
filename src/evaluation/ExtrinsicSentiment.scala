package evaluation

import experiments.Params
import models.EmbeddingModel
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingSequenceLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import transducer.AbstractLM
import utils.Tokenizer

import java.io.File
import scala.io.Source

class ExtrinsicSentiment(params: Params, tokenizer: Tokenizer, lm: AbstractLM) extends ExtrinsicPOS(params, tokenizer, lm) {

  var trainingSize = 10000
  var maxWindowSize = 100

  override def getClassifier(): String = "sentiment"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/sentiment/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/sentiment/test.txt"
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {

    Source.fromFile(filename).getLines().filter(l => l.contains("\t")).take(trainingSize).map(line => {
      val mline = line.toLowerCase(locale)
      val array = mline.split("\t")
      val sentence = array.take(array.length - 1).mkString(" ")
      val label = array.last
      (sentence, label)
    })

  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if (categories == null) {
      val allsamples = loadSamples(getTraining()).toArray
      categories = allsamples.map(_._2).toSet.toArray
      maxWindowSize = params.evalWindowLength //allsamples.map(_._1.split("\\s+").length).max

      println("Max window Size: " + maxWindowSize)
      println("Category Size: " + categories.length)
    }

    categories
  }

  def padBegining(array: Array[String], maxSize: Int): Array[String] = {
    val newArray = array.take(maxSize)
    val padSize = maxSize - newArray.length
    var padArray = newArray
    for (i <- 0 until padSize) {
      padArray = "zero" +: padArray
    }
    padArray
  }


  override def iterator(filename: String): MultiDataSetIterator = {

    new MultiDataSetIterator {

      var lines = loadSamples(filename)
      val labelArray = labels()

      override def next(i: Int): MultiDataSet = {
        var cnt = 0
        var trainStack = Array[INDArray]()
        var trainOutputStack = Array[INDArray]()
        var maskInputStack = Array[INDArray]()
        var maskOutputStack = Array[INDArray]()


        while (cnt < params.evalBatchSize && hasNext) {
          val (sentence, label) = lines.next()
          val tokens = padBegining(sentence.split("[\\s\\p{Punct}]+"), maxWindowSize)
            .map(_.trim).filter(_.nonEmpty)
          val sentenceVector = vectorize(tokens.map(token => forward(token)), maxWindowSize)
          val outputVector = onehot(labelArray.indexOf(label), labelArray.length)

          trainStack = trainStack :+ sentenceVector
          trainOutputStack = trainOutputStack :+ outputVector
          maskInputStack = maskInputStack :+ maskInput(maxWindowSize, tokens.length)
          maskOutputStack = maskOutputStack :+ maskOutput()
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
        lines = loadSamples(filename)
      }

      override def hasNext: Boolean = lines.hasNext

      override def next(): MultiDataSet = next(0)
    }

  }

  override def universe(): Set[String] = {
    Source.fromFile(getTraining()).getLines().filter(l => l.contains("\t"))
      .take(trainingSize)
      .flatMap(line => {
        val sentence = line.split("\t").head
        tokenizer.standardTokenizer(sentence)
      }).toSet
  }


  override def model(): ComputationGraph = {
    val categorySize = labels().length

    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .dataType(DataType.FLOAT)
      .activation(Activation.TANH)
      .updater(new Adam(params.lrate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(params.embeddingLength))
      .layer("input-lstm", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.hiddenLength)
        .activation(Activation.TANH).build, "input")
      .layer("attention", new SelfAttentionLayer.Builder().nOut(params.hiddenLength).nHeads(params.nheads).projectInput(true).build(), "input-lstm")
      .layer("pooling", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build(), "attention")
      .layer("dense_base", new DenseLayer.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength).activation(Activation.RELU).build(), "pooling")
      .layer("dense", new DenseLayer.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength).activation(Activation.RELU).build(), "dense_base")
      .layer("output", new OutputLayer.Builder().nIn(params.hiddenLength).nOut(categorySize).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "dense")

      .build()

    new ComputationGraph(conf)
  }

  override def train(filename: String): EmbeddingModel = {

    var i = 0
    val fname = params.modelEvaluationFilename()
    val modelFile = new File(fname)
    println("Self-Attention LSTM filename: " + fname)
    if (!(modelFile.exists()) || params.forceTrain) {


      val size = Source.fromFile(filename).getLines().size

      load()

      computationGraph = model()

      //val statsStorage = new InMemoryStatsStorage()
      //val uiServer = UIServer.getInstance()
      //uiServer.attach(statsStorage)

      computationGraph.addListeners(new PerformanceListener(1, true))

      val multiDataSetIterator = iterator(filename)

      var start = System.currentTimeMillis()
      var isTrained = false
      sampleCount = 0
      while (i < params.evalEpocs) {

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

}
