package evaluation

import sampling.experiments.SampleParams
import utils.{Params, Tokenizer}

import scala.io.Source

class ExtrinsicNER(params:SampleParams, tokenizer: Tokenizer) extends ExtrinsicPOS(params, tokenizer){



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
