package transducer

import experiments.Params
import utils.Tokenizer.locale

import java.io.File

class SkipLM(override val params: Params) extends AbstractLM(params) {


  val modelFilename = s"${parent}/resources/transducers/skip${params.lmID()}.bin"

  override def normalize(): AbstractLM = {
    lm.normalize()
    this
  }


  override def prune(): AbstractLM = {
    lm.prune(params.lmPrune)
    this
  }

  def train(sequence: Array[String]): SkipLM = {
    lm.countCombinatoric(sequence, params.lmSlideLength, params.lmSkip)
    this
  }

  def train(sequence: String): SkipLM = {
    lm.countCombinatoric(sequence.toCharArray.map(_.toString), params.lmSlideLength, params.lmSkip)
    this
  }

  def trainDictionary(item: String): SkipLM = {
    lm.transducer.addPrefix(item)
    this
  }

  def trainDictionary(item: Array[String]): SkipLM = {
    lm.transducer.addPrefixes(item)
    this
  }


  override def splitSentence(sentence: Array[String]): Array[String] = {
    lm.skipSlideLink(sentence, params.lmWindowLength, params.lmSkip, params.lmTopSplit)
  }


  override def splitToken(token: String): Array[String] = {
    lm.tokenSplit(token, params.lmTopSplit)
  }

  def getModelFilename(): String = {
    modelFilename
  }


  override def exists(): Boolean = {
    new File(getModelFilename()).exists()
  }

  override def load(transducer: Transducer): SkipLM = {
    if (exists()) {
      println("Loading filename: " + getModelFilename())
      lm = TransducerOp.loadLM(getModelFilename(), transducer)
    }
    else {
      println(s"SkipLM model: ${getModelFilename()} not found")
      lm = new TransducerLM(transducer)
    }

    this
  }

  override def load(): SkipLM = {
    load(lm.transducer)
    this
  }


  override def trainSentence(line: String): AbstractLM = {
    val arr = line.split("\\s+")
      .map(_.toLowerCase(locale))
      .filter(item => item.matches("\\p{L}+"))
      .sliding(params.windowLength, 1)
      .toArray

    arr.foreach(sequence => lm.countCombinatoric(sequence, params.lmSlideLength, params.lmSkip))
    this
  }

  override def loadTrain(): this.type = {
    if (exists()) {
      lm = TransducerOp.loadLM(modelFilename, lm.transducer)
      if (params.lmForceTrain || lm.isEmpty()) {
        lm = TransducerOp.trainParallelCombinatoricBySkip(lm, textFilename, modelFilename, params)
        TransducerOp.saveLM(modelFilename, lm)
      }
    }
    else {
      lm = TransducerOp.trainParallelCombinatoricBySkip(lm.transducer, textFilename, modelFilename, params)
      TransducerOp.saveLM(modelFilename, lm)
    }

    this
  }

  override def copy(): AbstractLM = {
    val skipLM = new SkipLM(params)
    skipLM.lm = lm.copy()
    this
  }

  override def save(): AbstractLM = {
    TransducerOp.saveLM(modelFilename, lm)
    this
  }
}
