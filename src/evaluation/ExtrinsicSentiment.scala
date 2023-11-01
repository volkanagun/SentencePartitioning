package evaluation

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingSequenceLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import sampling.experiments.SampleParams
import utils.Params

import scala.io.Source

class ExtrinsicSentiment(params:SampleParams) extends ExtrinsicLSTM(params){

  var categories :Array[String] = null
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

    Source.fromFile(filename).getLines().map(line=> {
      val Array(p1, p2) = line.split("\t")
      (p1, p2)
    })
  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if(categories == null){
      categories = loadSamples(getTraining()).map(_._2).toSet.toArray
    }

    categories
  }



}
