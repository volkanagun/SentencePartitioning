package models

import experiments.Params
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingSequenceLayer, GlobalPoolingLayer, LSTM, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import transducer.AbstractLM
import utils.Tokenizer

class GravesLSTMModel(params:Params, tokenizer:Tokenizer,  lm:AbstractLM) extends SelfAttentionLSTM(params, tokenizer, lm) {

  override def model(dictionarySize:Int): ComputationGraph = {


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
      .layer("output-lstm", new LSTM.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength)
        .activation(Activation.TANH).build, "input-lstm")
      .layer("last-lstm", new LastTimeStep(new LSTM.Builder().nIn(params.hiddenLength).nOut(params.hiddenLength)
        .activation(Activation.RELU).build), "output-lstm")
      .layer("output", new OutputLayer.Builder().nIn(params.hiddenLength).nOut(dictionarySize).activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).build, "last-lstm")
      .setInputTypes(InputType.recurrent(dictionarySize))
      .build()

    conf.setTrainingWorkspaceMode(WorkspaceMode.ENABLED)
    conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED)

    new ComputationGraph(conf)
  }

}
