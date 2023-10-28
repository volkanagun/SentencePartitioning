package evaluation

import models.EmbeddingModel

trait ExtrinsicFunction extends IntrinsicFunction {

  def train(filename:String):this.type

}
