package transducer

import experiments.Params
import utils.Tokenizer.locale

import java.io.File

/**
 * @Author Dr. Hayri Volkan Agun
 * @Date 21.03.2022 22:56
 * @Project BigLanguage
 * @Version 1.0
 */

class SyllableLM(override val params:Params)  extends AbstractLM(params) {


  lm.transducer = TransducerOp.fromSyllables()
  val modelFilename = s"${parent}/resources/transducers/syllables${params.lmID()}.bin"

  var notAcceptRules = TransducerOp.syllableNotAccept().map(_.r)


  val syllableSplit: (String => Array[String]) = (input: String) => {
    val result = lm.transducer.tokenSplit(input, params.lmTopSplit)
      .filter(_.nonEmpty).filter(syllables=>{
        !notAcceptRules.exists(regex=> regex.findAllIn(syllables).nonEmpty)
      })

    if(result.isEmpty) result :+ input
    else result
  }

  override def initialize(): SyllableLM.this.type = {
    lm = new TransducerLM(TransducerOp.fromSyllables())


    val seqTransducer = lm.seqTransducer
    TransducerOp.fromDictionaryByInfer(syllableSplit, seqTransducer, dictionaryTextFilename, params)
    TransducerOp.fromTextByInfer(syllableSplit, seqTransducer, textFilename, params)

    TransducerOp.saveLM(modelFilename, lm)

    this
  }


  def accept(item:String):Boolean={
    !notAcceptRules.exists(r=> r.findAllIn(item).nonEmpty)
  }

  def exists(): Boolean = {
    new File(modelFilename).exists()
  }

  override def load(): SyllableLM = {
    if (exists()) lm = TransducerOp.loadLM(modelFilename)
    else {
      lm = new TransducerLM(lm.transducer)
    }
    this
  }

  override def load(transducer: Transducer): SyllableLM = {
    if (exists()) lm = TransducerOp.loadLM(modelFilename)
    else {
      lm = new TransducerLM(transducer)
    }

    this
  }

  override def train(sequence: Array[String]): AbstractLM = {
    lm.countCombinatoric(sequence,params.lmTopSplit, params.lmSlideLength)
    this
  }

  override def train(sequence: String): AbstractLM = {
    lm.countCombinatoric(sequence.toCharArray.map(_.toString), params.lmTopSplit, params.lmSlideLength)
    this
  }

  override def trainDictionary(item: String): AbstractLM = {
    lm.transducer.addPrefix(item)
    this
  }

  override def trainDictionary(item: Array[String]): AbstractLM = {
    lm.transducer.addPrefixes(item)
    this
  }

  override def splitSentence(sentence: Array[String]): Array[String] = lm.slideSplit(syllableSplit, sentence, params.lmSlideLength, params.lmTopSplit)
  override def splitToken(token: String): Array[String] = syllableSplit(token).take(params.lmTopSplit)

  override def normalize(): AbstractLM = {
    lm.normalize()
    this
  }
  override def prune(): AbstractLM = {
    lm.prune(params.lmPrune)
    this
  }

  override def trainSentence(sentence: String): AbstractLM = {
    val arr = sentence.split("\\s+")
      .map(_.toLowerCase(locale))
      .filter(item => item.matches("\\p{L}+"))
      .sliding(params.lmWindowLength)

    arr.toArray.foreach(lm.count(_, params.lmTopSplit, params.lmSlideLength))
    this

  }

  override def loadTrain(): SyllableLM = {
    if (exists()) {
      lm = TransducerOp.loadLM(modelFilename)
      if(params.lmForceTrain || lm.isEmpty()) {
        lm = TransducerOp.trainParallel(lm, textFilename, modelFilename, params)
        TransducerOp.saveLM(modelFilename, lm)
      }
    }
    else {
      lm = TransducerOp.trainParallel(lm.transducer, textFilename, modelFilename, params)
      TransducerOp.saveLM(modelFilename, lm)
    }

    this
  }

  def find(samples:Array[String]):Array[Array[String]]={
    samples.map(sample=>lm.infer(sample, params.lmTopSplit))
  }



  override def copy(): AbstractLM = {
    val syllableLM = new SyllableLM(params)
    syllableLM.lm = lm.copy()
    this
  }

  override def save(): AbstractLM = {

    TransducerOp.saveLM(modelFilename, lm)
    this
  }
}
