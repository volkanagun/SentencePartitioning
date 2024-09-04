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

class LMRankEfficient(params: Params) extends RankLM(params) {

  override val modelFilename = s"${parent}/resources/transducers/rank-efficient${params.lmID()}.bin"

  override def getModelFilename(): String = modelFilename


/*
  override def partition(sentence: Array[String]): Array[Array[String]] = {
    var spanList = Array[WordSpan]()
    var start = 0
    for (i <- 0 until sentence.length) {
      spanList = spanList :+ new WordSpan(start, start + sentence(i).length)
        .setValue(sentence(i))
      start = start + sentence(i).length + 1
    }
    val text = sentence.mkString(" ")
    val lemmaList = wordLemmatizer.lemmatize(spanList, text)
    val random = new Random(17)
    val lemmaString = lemmaList.map(wordGroup => {
      val splittings = wordGroup.lemmaSplitList(lm.transducer.marker).distinct
      val sampled = random.shuffle(splittings.toSeq).take(params.lmSample).toArray
      sampled
    })
    lemmaString
  }
*/



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


}
