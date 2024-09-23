package models

import evaluation.{EvalScore, InstrinsicEvaluationReport}
import experiments.Params
import transducer.AbstractLM
import utils.Tokenizer

import java.io.{File, FileOutputStream, ObjectOutputStream}
import scala.io.Source

class FastTextModel(params:Params, tokenizer: Tokenizer,  lm:AbstractLM) extends CBOWModel(params, tokenizer, lm) {


  var mydictionary = Map[String, Array[Float]]()
  override def train(filename: String): EmbeddingModel = {

    val fname = params.embeddingsFilename()

    if (!(new File(fname).exists())|| params.forceTrain) {
      println("FastText filename: " + fname)
      val vocabulary = Source.fromFile(filename).getLines().flatMap(line=> line.split("\\s+")).toSet
      val lines = Source.fromFile(params.fastTextBin).getLines()
      lines.next()
      while(lines.hasNext){
        val lineSplit = lines.next().split("\\s+")
        val word = lineSplit.head.toLowerCase(locale)
        if(vocabulary.contains(word)){
          val vector = lineSplit.tail.map(_.toFloat)
          mydictionary = mydictionary.updated(word, vector)
        }
      }
      save()
    }

    this
  }

  override def save(): EmbeddingModel = {
    val filename = params.embeddingsFilename()
    val printer = new ObjectOutputStream(new FileOutputStream(filename))
    printer.writeInt(mydictionary.size)
    mydictionary.foreach(pair=>{
      printer.writeObject(pair._1)
      printer.writeObject(pair._2)
      update(pair._1, pair._2)
    })
    printer.close()
    this
  }

  override def evaluate(model: EmbeddingModel): EvalScore = EvalScore(0d, 0d)

  override def setDictionary(set: Set[String], model: EmbeddingModel): FastTextModel.this.type = this

  override def setWords(set: Set[String]): FastTextModel.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): FastTextModel.this.type = this

  override def count(): Int = 0

  override def getClassifier(): String = "SkipGram"

  override def filter(group: Array[String]): Boolean = true

  override def evaluateReport(model: EmbeddingModel, embedParams: Params): InstrinsicEvaluationReport = new InstrinsicEvaluationReport()
}
