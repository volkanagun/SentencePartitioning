package evaluation

import experiments.Params
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import transducer.AbstractLM
import utils.Tokenizer

import scala.io.Source
import scala.util.Random

class ExtrinsicNER(params:Params, tokenizer: Tokenizer,  lm:AbstractLM) extends ExtrinsicPOS(params, tokenizer, lm){



  override def getClassifier(): String = "ner"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/ner/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/ner/test.txt"
  }


  override def universe(): Set[String] = {
    val source = Source.fromFile(getTraining(), "UTF-8")
    try source.getLines().filter(_.trim.nonEmpty).map(_.split("\\s+").head).toSet
    finally source.close()
  }

  private def documents(filename: String): Vector[String] = {
    val source = Source.fromFile(filename, "UTF-8")
    try source.mkString.split("(?:\\r?\\n){2,}").map(_.trim).filter(_.nonEmpty).toVector
    finally source.close()
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {
    val rnd = new Random(17)
    rnd.shuffle(documents(filename)).iterator.map(document => {
      val sentence = document.linesIterator.filter(_.trim.nonEmpty).map(wordLabel => {
        val fields = wordLabel.trim.split("[\t\\s]+")
        fields.head + "/" + fields.last
      }).mkString(" ").toLowerCase(locale)
      (sentence, "")
    })
  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if (categories == null) {
      println("Finding category labels")
      categories = Source.fromFile(getTraining()).getLines().map(line=> line.toLowerCase(locale))
        .map(item => item.split("[\\t\\s]+").last)
        .toSet.toArray
      categories ="NONE" +: categories
      categories
    }
    else {
      categories
    }
  }



}
