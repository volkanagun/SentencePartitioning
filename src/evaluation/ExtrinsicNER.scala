package evaluation

import experiments.Params
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import utils.Tokenizer

import scala.io.Source
import scala.util.Random

class ExtrinsicNER(params:Params, tokenizer: Tokenizer) extends ExtrinsicPOS(params, tokenizer){



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
    Source.fromFile(getTraining()).getLines().map(token => {
      val word = token.split("\\s+").head
      word
    }).toSet
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {

    val rnd = new Random(17)
    val array = Source.fromFile(filename).getLines().toSeq
    var sentences  = Array[(String, String)]()
    var sentence = ""
    rnd.shuffle(array).iterator.foreach(wordLabel => {
      if(wordLabel.trim.nonEmpty){
        val Array(word, label) = wordLabel.split("[\t\\s]+")
        sentence += " " + word+"/"+label
      }
      else{
        val crrSentence = sentence.trim.toLowerCase(locale)
        sentences = sentences :+ (crrSentence, "")
        sentence = ""
      }
    })

    sentences.iterator
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
