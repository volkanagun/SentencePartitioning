package transducer

import experiments.Params

class SuffixLM(params:Params) extends RankLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/suffix${params.lmID()}.bin"
  override def findMinSplitSentence(sentence: Array[String]): Array[String] = {
    lm.suffixRank(sentence, params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = {
    lm.suffixRank(sentence, params.lmSkip, params.lmTopSplit, params.lmIterations)
  }
}

object SuffixLM{

  def apply(windowSize:Int, skipSize:Int, topSplit:Int):SuffixLM={
    val p = new Params()
    p.lmWindowLength = windowSize
    p.lmSkip = skipSize
    p.lmTopSplit = topSplit
    p.lmTrainDictionary = true
    new SuffixLM(p)
  }

  def test1(suffixLM: SuffixLM): Unit = {
    Array("adama yardım eden şahıs mahkemeye çıkarıldı",
      "geçen iftarda da top atıldı",
      "zaman zaman yağmur yağıyordu").foreach(sentence=>{
          suffixLM.findMinSplitSentence(sentence.split("\\s"))
            .foreach(item=> println(item))
    })
  }

  def main(args: Array[String]): Unit = {
    val suffixLM = SuffixLM(3, 2, 3)
    suffixLM.load().test()
    test1(suffixLM)
  }
}
