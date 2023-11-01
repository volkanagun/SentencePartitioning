package evaluation

import sampling.experiments.SampleParams
import utils.Params

import scala.io.Source

class ExtrinsicNER(params:SampleParams) extends ExtrinsicPOS(params){



  override def getClassifier(): String = "ner"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/ner/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/ner/test.txt"
  }




}
