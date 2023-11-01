package evaluation

import org.deeplearning4j.nn.api.layers.RecurrentLayer
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sampling.experiments.SampleParams
import utils.Params

import scala.io.Source

class ExtrinsicPOS(params: SampleParams) extends ExtrinsicLSTM(params) {

  var categories: Array[String] = null

  override def getClassifier(): String = "pos"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/pos/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/pos/test.txt"
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {
    Source.fromFile(filename).getLines().map(line => {
      val input = line.split("\t").head
      (input, input)
    })
  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if (categories == null) {
      println("Finding category labels")
      categories = Source.fromFile(getTraining()).getLines()
        .flatMap(item => item.split("\\s").map(_.split("/").last).distinct)
        .toSet.toArray
      categories
    }
    else {
      categories
    }
  }

  override def iterator(filename: String): MultiDataSetIterator = {
    new MultiDataSetIterator {

      var samples = loadSamples(filename)
      val categorySize = labels().length

      override def next(i: Int): MultiDataSet = {


        var inputLeftStack = Array[INDArray]()
        var inputRightStack = Array[INDArray]()
        var outputStack = Array[INDArray]()
        var i = 0;
        while (i < params.batchSize && samples.hasNext) {
          val (input, output) = samples.next()
          val tokens = input.split("\\s+")
          val inputOutput = tokens.map(word => word.split("/"))

          val inputLeft = inputOutput.map(_.head).take(params.evalWindowLength)
          val inputLeftArray = inputLeft.map(ngram => {
            retrieve(ngram)
          })

          val inputLeftIndex = index(inputLeftArray)
          val inputRightIndex = index(inputLeftArray.reverse)
          val outputIndices = inputOutput.map(_.last)
            .map(output => labels().indexOf(output))
            .take(params.evalWindowLength)

          val outputArray = onehot(outputIndices, categorySize)


          inputLeftStack :+= inputLeftIndex
          inputRightStack :+= inputRightIndex
          outputStack :+= outputArray
          i = i + 1
        }

        val vLeft = Nd4j.vstack(inputLeftStack: _*)
        val vRight = Nd4j.vstack(inputRightStack: _*)
        val vOutput = Nd4j.vstack(outputStack: _*)


        new MultiDataSet(Array(vLeft, vRight), Array(vOutput))
      }

      override def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor): Unit = ???

      override def getPreProcessor: MultiDataSetPreProcessor = ???

      override def resetSupported(): Boolean = true

      override def asyncSupported(): Boolean = false

      override def reset(): Unit = {
        samples = loadSamples(filename)
      }

      override def hasNext: Boolean = samples.hasNext

      override def next(): MultiDataSet = next(0)
    }
  }

  override def model(): ComputationGraph = {

    val categorySize = labels().length
    val conf = new NeuralNetConfiguration.Builder()
      .cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
      .updater(new Adam.Builder().learningRate(params.lrate).build())
      .dropOut(0.5)
      .graphBuilder()
      .allowDisconnected(true)
      .addInputs("left", "right")
      .addVertex("stack", new org.deeplearning4j.nn.conf.graph.StackVertex(), "left", "right")
      .addLayer("embedding", new EmbeddingSequenceLayer.Builder().inputLength(params.evalWindowLength)
        .nIn(params.dictionarySize).nOut(params.embeddingLength).build(),
        "stack")
      .addVertex("leftemb", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "embedding")
      .addVertex("rightemb", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "embedding")
      //can use any label for this
      .addLayer("leftout", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.hiddenLength)
        .activation(Activation.RELU)
        .build(), "leftemb")
      .addLayer("rightout", new LSTM.Builder().nIn(params.embeddingLength).nOut(params.hiddenLength)
        .activation(Activation.RELU)
        .build(), "rightemb")
      .addVertex("merge", new MergeVertex(), "leftout", "rightout")
      .addLayer("output-lstm", new LSTM.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength)
        .activation(Activation.RELU)
        .build(), "merge")
      .addLayer("output",
        new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
          .activation(Activation.SOFTMAX)
          .nOut(categorySize)
          .dataFormat(RNNFormat.NWC)
          .build(), "output-lstm")
      .setOutputs("output")
      .setInputTypes(InputType.recurrent(params.dictionarySize),
        InputType.recurrent(params.dictionarySize))
      .build()

    val graph = new ComputationGraph(conf)

    updateWeights(graph)


  }
}
