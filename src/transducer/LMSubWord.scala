package transducer

import experiments.Params
import tagging.lemmatizer.{WordGroup, WordSpan}
import transducer.TransducerOp.locale
import utils.Tokenizer

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source
import scala.util.Random

class LMSubWord(params: Params) extends SkipLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/subword${params.lmID()}.bin"

  override def getModelFilename(): String = modelFilename

  override def splitSentence(sentence: Array[String]): Array[String] = {
    lm.pageSlideEfficientRank(sentence, params.lmWindowLength, params.lmSkip, params.lmTopSplit, params.lmIterations)
  }

  override def splitToken(token: String): Array[String] = {
    lm.tokenEntropySplit(token, params.lmTopSplit, params.lmSample)
  }

  override def exists(): Boolean = {
    new File(getModelFilename()).exists()
  }

  override def loadTrain(): this.type = {
    if (exists()) {
      lm = TransducerOp.loadLM(getModelFilename(), lm.transducer)
      if (params.lmForceTrain || lm.isEmpty()) {
        lm = TransducerOp.trainEfficientCombinatoricBySkip(lm, params.sentencesFile,getModelFilename(),  params)
        TransducerOp.saveLM(getModelFilename(), lm)
      }
    }
    else {
      lm = TransducerOp.trainEfficientCombinatoricBySkip(lm, textFilename, getModelFilename(), params)
      TransducerOp.saveLM(getModelFilename(), lm)
    }

    this

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

}
