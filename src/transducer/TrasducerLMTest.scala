package transducer

import experiments.Params
import transducer.LemmaLM.lemmaLM

object TrasducerLMTest {


  def params(): Params = {
    val params = new Params()
    params.lmTrainDictionary = true
    params.lmMaxSentence = 1000
    params.lmEpocs = 24
    params
  }

  def trainSequences(): Array[Array[String]] = {
    Array(
      Array("yaşamdan", "ve", "sanattan", "anlamayanların", "topluluğudur", ".")
    )
  }

  def testSequences(): Array[Array[String]] = {
    Array(
      Array("yaşam", "sanatı")
    )
  }

  def testLemmaLM(): Unit = {
    val sequence = trainSequences()
    val testSequence = testSequences()
    lemmaLM = new LemmaLM(params).loadTrain()

    sequence.foreach(items => {
      lemmaLM.trainDictionary(items); lemmaLM.train(items)
    })
    lemmaLM.prune().normalize()

    testSequence.foreach(items => {
      val subToken = lemmaLM.subsequence(items)
      println(subToken)
    })
  }

  def testNGramLM(): Unit = {
    val sequence = trainSequences()
    val testSequence = testSequences()
    lemmaLM = new LemmaLM(params).loadTrain()

    sequence.foreach(items => {
      lemmaLM.trainDictionary(items); lemmaLM.train(items)
    })
    lemmaLM.prune().normalize()

    testSequence.foreach(items => {
      val subToken = lemmaLM.findMinSplitSentence(items)
      println(subToken.mkString(" "))
    })
  }
}
