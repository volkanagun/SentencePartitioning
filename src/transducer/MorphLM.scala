package transducer

import experiments.Params

class MorphLM(trainFilename:String, params: Params) extends RankLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/morph${params.lmID()}.bin"

  override def initialize(): this.type = {
    if (!exists()) {
      lm.transducer = TransducerOp.fromMorphology(lm.transducer, trainFilename, params)
      lm = new TransducerLM(lm.transducer)
      save()
      System.gc()
      this
    }
    else {
      load()
      this
    }
  }

  def analyze(sentence:Array[String]):Array[Array[String]]={
    val tokens = findMinSplitSentence(sentence)
    val split = lm.transducer.split
    tokens.map(token=>{
      lm.transducer.produceOutput(token.split(split))
    })
  }
}
