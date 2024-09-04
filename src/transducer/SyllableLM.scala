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


  override def initialize(): SyllableLM.this.type = {
    lm = new TransducerLM(TransducerOp.fromSyllables())
    val infer: (String => Array[String]) = (input: String) => {
      lm.transducer.multipleSplitSearch(input, params.lmTopSplit)
        .flatMap(seq => seq.split(lm.transducer.split)).map(_.trim)
        .filter(_.nonEmpty).filter(syllables=>{
          !notAcceptRules.exists(regex=> regex.matches(syllables))
        })
    }

    val seqTransducer = lm.seqTransducer
    TransducerOp.fromDictionaryByInfer(infer, seqTransducer, dictionaryTextFilename, params)
    TransducerOp.fromTextByInfer(infer, seqTransducer, textFilename, params)

    TransducerOp.saveLM(modelFilename, lm)

    this
  }

  override def subsequence(sentence: Array[String]): String = {
    findMinSplitSentence(sentence)
      .flatMap(item => item.split(lm.transducer.split)
        .filter(_.nonEmpty))
      .mkString(" ")
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

  override def findMinSplit(token: String): Array[String] = lm.inferMinSplit(token, params.lmTopSplit)

  override def findMinSplitEfficient(sentence: Array[String]): Array[String] = lm.inferMinEfficientSplit(sentence, notAcceptRules, params.lmTopSyllableSplit, params.lmTopSplit)

  override def findMinSplitSentence(sentence: Array[String]): Array[String] = lm.inferMinTokenSplit(sentence, notAcceptRules, params.lmTopSyllableSplit, params.lmTopSplit)
  override def findLikelihoodSentence(sentence: Array[String]): Array[String] = lm.inferMinTokenSplit(sentence, notAcceptRules, params.lmTopSyllableSplit, params.lmTopSplit)
  override def findMultiSplitSentence(sentence: Array[String]): Array[String] = lm.inferMultiTokenSplit(sentence, params.lmTopSplit)
  override def findMinSlideSplitSentence(sentence: Array[String]): Array[String] = lm.inferSlideMinTokenSplit(sentence, params.lmSlideLength, params.lmTopSplit)

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

object SyllableLM{

  def apply(windowSize:Int, topSplit:Int):SyllableLM = {
    val params = new Params()
    params.lmWindowLength = windowSize
    params.lmTopSplit = topSplit
    val syllableLM = new SyllableLM(params)
      .load()
    //syllableLM.test()
    syllableLM
  }

  def testSmall(): Unit ={

    val params = new Params()
    params.lmTrainDictionary = true
    params.lmEpocs = 100
    params.lmMaxSentence = 10000

    val syllableLM = new SyllableLM(params)
      .load()
      .initialize()
    //syllableLM.loadTrain()

    syllableLM.findLikelihoodSentence(Array("ihlal")).foreach(result => println(result.mkString("")))

  }

  def main(args: Array[String]): Unit = {
    testSmall()
  }

}