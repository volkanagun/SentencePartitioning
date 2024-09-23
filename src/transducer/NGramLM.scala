package transducer

import experiments.Params
import utils.Tokenizer

import java.io.File


class NGramLM(override val params: Params) extends AbstractLM(params) {

  val modelFilename = s"${parent}/resources/transducers/ngrams${params.lmID()}.bin"


  override def copy(): NGramLM = {
    val ngramLM = new NGramLM(params)
    ngramLM.lm = lm.copy()
    ngramLM
  }

  override def normalize(): AbstractLM = {
    lm.normalize()
    this
  }

  def getModelFilename(): String = {
    modelFilename
  }

  override def prune(): AbstractLM = {
    lm.prune(params.lmPrune)
    this
  }


  override def save(): NGramLM.this.type = {

    TransducerOp.saveLM(getModelFilename(), lm)

    this
  }



  override def exists(): Boolean = {
    new File(getModelFilename()).exists()
  }

  override def load(input: Transducer): NGramLM = {

    if (exists()) {
      lm = TransducerOp.loadLM(getModelFilename(), input)

    }
    else {
      lm = new TransducerLM(input)

    }

    this
  }

  override def load(): NGramLM = {
    load(new Transducer())
    this
  }


  override def trainSentence(sentence: String): NGramLM = {
    val arr = sentence.split("\\s+")
      .map(_.toLowerCase(Tokenizer.locale))
      .filter(item => item.matches("\\p{L}+"))
      .sliding(params.lmWindowLength, 1)
      .toArray

    arr.foreach(sequence => lm.countCombinatoric(sequence, params.lmTopSplit, params.lmSlideLength))
    this
  }

  override def loadTrain(): NGramLM = {

    if (exists()) {

      lm = TransducerOp.loadLM(modelFilename, new Transducer())

      if(params.lmForceTrain || lm.isEmpty()) {
        lm = TransducerOp.trainParallelCombinatoricBySlide(lm, textFilename, modelFilename, params)

        TransducerOp.saveLM(modelFilename, lm)
      }

    }
    else {
      lm = TransducerOp.trainParallelCombinatoricBySlide(new Transducer(), textFilename, modelFilename, params)
      TransducerOp.saveLM(modelFilename, lm)
    }

    this
  }

  override def train(sequence: Array[String]): NGramLM = {
    lm.countCombinatoric(sequence, params.lmTopSplit, params.lmSlideLength)
    this
  }

  override def train(sequence: String): NGramLM = {
    lm.count(sequence, params.lmTopSplit, params.lmSlideLength)
    this
  }

  override def trainDictionary(item: String): NGramLM = {
    lm.transducer.addPrefix(item)
    this
  }

  override def trainDictionary(item: Array[String]): NGramLM = {
    lm.transducer.addPrefixes(item)
    this
  }


  override def splitSentence(sentence: Array[String]): Array[String] = lm.slideSplit(sentence, params.lmSlideLength, params.lmTopSplit)

  override def splitToken(token: String): Array[String] = lm.tokenSplit(token, params.lmTopSplit)
}

