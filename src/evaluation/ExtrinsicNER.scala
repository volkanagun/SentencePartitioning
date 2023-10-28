package evaluation

import utils.Params

import scala.io.Source

class ExtrinsicNER(params:Params) extends ExtrinsicPOS(params){



  override def getClassifier(): String = "NER"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/ner/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/ner/test.txt"
  }




}
