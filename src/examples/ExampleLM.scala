package examples

import experiments.Params
import transducer.RankLM

object ExampleLM {

  def createParams():Params={
    val parameters = new Params()

    parameters.adapterName = "rankLM"
    parameters.sentencesFile = "resources/text/sentences/sentences-tr.txt"
    parameters.lmWindowLength = 2
    parameters.lmEpocs = 1000
    parameters.lmMaxSentence = 2400
    parameters.lmMaxSentenceLength = 120
    parameters.lmTopSplit = 10
    parameters.lmPrune = Int.MaxValue
    parameters.lmThreads = 480

    parameters
  }

  def loadModel():RankLM={
    val parameters = createParams()
    val rankLM = new RankLM(parameters)
    rankLM.initialize().load()
    rankLM
  }

  def trainModel() : RankLM={
    val parameters = createParams()
    val rankLM = new RankLM(parameters).initialize()
      .loadTrain()

    rankLM.asInstanceOf[RankLM]

  }

  def exampleSentence(): Unit = {

    val rankLM = trainModel()
    val sentences = Array[String](
      "yaşamın ucuna yolculuk filmi gerçekten çok güzeldi .",
      "saat ondan sonra burada bulunan alarm sistemi devreye giriyor .",
      "başbakanın sözleri çok konuşuldu .",
      "örneklerin elden geçmesi gerekiyordu ."
    )

    sentences.foreach(sentence=>{
      val tokens = sentence.split("\\s")
      val partitions = rankLM.splitSentence(tokens)
      val partitionString = partitions.mkString(" ")
      println(partitionString)
    })
  }

  def exampleToken(): Unit = {

    val rankLM = loadModel()
    val sentences = Array[String](
      "giriyor",
      "çok",
      "saat"
    )

    sentences.foreach(sentence=>{
      val tokens = sentence.split("\\s")
      tokens.foreach(token=>{
        val splits =rankLM.splitToken(token)
        splits.foreach(split=>{
          println("Token: " + token + " Split: "+split)
        })
      })
    })
  }

  def main(args: Array[String]): Unit = {
    exampleSentence()
  }
}
