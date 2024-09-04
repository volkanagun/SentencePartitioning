package transducer

import experiments.Params

class RankLM(params: Params) extends SkipLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/rank${params.lmID()}.bin"

  override def getModelFilename(): String = modelFilename


  override def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    lm.pageRank(sentence,  params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = {
    lm.pageRankEfficient(sentence, params.lmSample,  params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = {
    lm.pageRank(sentence, params.lmSkip,params.lmTopSplit, params.lmIterations)
  }

  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = {
    lm.pageSlideRank(sentence, params.lmSlideLength, params.lmSkip, params.lmTopSplit, params.lmIterations)
  }


  override def copy(): AbstractLM = {
    val rankLM = new RankLM(params)
    rankLM.lm = lm.copy()
    rankLM
  }

}

object RankLM {

  def apply(windowSize:Int, maxSkip:Int, topSplit:Int):RankLM={
    val params = new Params()
    params.lmSlideLength = 4
    params.lmWindowLength= windowSize
    params.lmSkip = maxSkip
    params.lmTopSplit = topSplit
    val rankLM = new RankLM(params)
    rankLM.load()
    rankLM.test()
    rankLM
  }

  def test3(): Unit = {

    val params = new Params()
    params.lmTrainDictionary = true
    params.lmepocs = 1
    params.lmMaxSentence = 1000
    val rankingLM = new RankLM(params).loadTrain()
    val array1 = "zaman çok".toCharArray.map(_.toString)
    val result1 = rankingLM.findMinSplitSentence(array1)
    println("Sentence: " + array1.mkString(" ") + "\n" + result1.mkString(" "))

  }

  def main(args: Array[String]): Unit = {
    test3()
  }
}
