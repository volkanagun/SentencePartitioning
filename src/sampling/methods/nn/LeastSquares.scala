package sampling.methods.nn

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.regression.{LinearModel, OLS}

class LeastSquares(val computeMod: Int, dictionarySize: Int, embeddingSize: Int) extends InstanceScorer(dictionarySize, embeddingSize) {

  var model: LinearModel = null
  var samples = Array[Array[Double]]()


  def compute(): this.type = {

    try {
      val data = samples.map(vector => random.between(-0.05, 0.05) +: vector)
      val items = "y" +: Range(0, embeddingSize).map(i => "x" + i.toString())
      val df = DataFrame.of(data, items: _*)
      model =OLS.fit(Formula.lhs("y"), df)
    }
    catch {
      case (e: Throwable) => {
        println("Matrix is singular.")
        e.printStackTrace()
      }
    }

    this
  }

  override def score(instance: TextInstance): Double = {
    val vector = random.between(-0.05, 0.05) +: embeddingVector(instance.featureSequence)
    val result = math.abs(model.predict(vector))
    result
  }

  override def add(instance: TextInstance): LeastSquares.this.type = {
    samples = samples :+ embeddingVector(instance.featureSequence)
    count = count + 1
    this
  }
  override def postCompute(): LeastSquares.this.type = {
    compute()
  }

  override def init(instances: Array[TextInstance]): LeastSquares.this.type = {
    instances.foreach(instance => add(instance))
    compute()
    this
  }
}
