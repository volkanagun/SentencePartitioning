package models

import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

object DL4JGpu {

  def configure(): Unit = {
    Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT)
    println("ND4J backend: " + Nd4j.getBackend.getClass.getName)
    println("ND4J executioner: " + Nd4j.getExecutioner.getClass.getName)
  }

  def prepare(graph: ComputationGraph): ComputationGraph = {
    configure()
    graph.init()
    graph
  }
}
