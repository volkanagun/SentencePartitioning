package evaluation

import utils.Params

import scala.io.Source

class ExtrinsicSentiment(params:Params) extends ExtrinsicLSTM(params){

  var categories :Array[String] = null
  override def getClassifier(): String = "SENTIMENT"

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
