package evaluation

import models.EmbeddingModel
import org.json4s.DefaultFormats
import org.json4s.JsonAST.JValue
import org.json4s.jackson.JsonMethods._
import sampling.experiments.SampleParams
import utils.Params

import java.io.{File, PrintWriter}
import java.util.concurrent.ForkJoinPool
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.collection.parallel.ForkJoinTaskSupport
import scala.io.Source

/**
 * @author Volkan Agun
 */
class IntrinsicEvaluation(val reportFilename: String) extends IntrinsicFunction with Serializable {

  var functions: Array[IntrinsicFunction] = Array()
  var testWords = Set[String]()
  var testEmbeddings = Set[(String, Array[Float])]()

  override def setDictionary(set: Set[String], model:EmbeddingModel): this.type = {

    println("Evaluating embeddings of the dictionary words")
    testWords = set
    testEmbeddings = testWords.toArray.par.map(target => (target, model.forward(target))).toArray.toSet

    functions.foreach(ier=> ier.setWords(testWords).setEmbeddings(testEmbeddings))
    this
  }


  override def setWords(set: Set[String]): IntrinsicEvaluation.this.type = {
    this.testWords = set
    this
  }

  override def setEmbeddings(set: Set[(String, Array[Float])]): IntrinsicEvaluation.this.type = {
    this.testEmbeddings = set
    this
  }

  override def getClassifier(): String = "main"

  override def count(): Int = functions.map(_.count()).sum


  override def universe(): Set[String] = {
    println("Computing set of queries")
    val set = functions.flatMap(function => function.universe()).toSet
    println("Queries are found...")
    set
  }

  override def evaluate(model: EmbeddingModel): EvalScore = {
    val evals = functions.par.map(_.evaluate(model))
    val tp = evals.map(_.tp).sum / evals.length
    val similarity = evals.map(_.similarity).sum / evals.length
    EvalScore(tp, similarity)
  }


  override def evaluateReport(model: EmbeddingModel, params:SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport()
    println(s"Total Evaluation Functions: ${functions.map(_.count()).sum}")
    val parFunctions= functions.par
    parFunctions.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(params.nthreads))
    parFunctions.map(intr => intr.evaluateReport(model, params)).toArray.foreach(intr => {
      ier.append(intr)
    })

    new File(reportFilename.substring(0, reportFilename.lastIndexOf("/"))).mkdirs()
    val pw = new PrintWriter(reportFilename)
    ier.setTrainTime(model.getTrainTime())
    ier.printXML(pw, params.toShortXML())
    pw.close()
    ier
  }


  override def filter(groups: Array[String]): Boolean = {
    functions.foreach(pred=> pred.filter(groups))
    functions.exists(pred => pred.filter(groups))
  }


  def compile(): this.type = {
    this
  }


  def attachEvaluations(jsonFilename: String): this.type = {
    implicit val formats = DefaultFormats
    val text = Source.fromFile(jsonFilename, "UTF-8").getLines().mkString("\n")

    val json = parse(text)
    (json \ "intrinsic").children.map(jobject => {
      attachEvaluation(jobject)
    })
    this
  }

  def attachEvaluation(jobject: JValue): Unit = {
    implicit val formats = DefaultFormats
    val labelOpt = (jobject \ "label").extractOpt[String]

    labelOpt match {
      case None => {}
      case Some(label) => {

        if (!IntrinsicEvaluationTypes.cosineSimilarity.r.findAllMatchIn(label).isEmpty) {
          val semeval = (jobject \ "semeval").extractOpt[String]
          val semanticclassifier = (jobject \ "classifier").extract[String]
          (semeval, semanticclassifier) match {
            case (Some(id), task) => {
              val pairs = (jobject \ "texts").children.map(_.extract[String]).map(comparison => {
                val Array(w1, w2) = comparison.split("[\\:\\+]")
                (w1, w2)
              }).toArray

              functions :+= SemEvalAnalogy(task, id, pairs)
              //functions :+= SemEvalAnalogyOrder(task, id, pairs)
            }
            case (None, _) => {}
          }

          val wordSource = (jobject \ "textA").extractOpt[String]
          val classifier = (jobject \ "classifier").extractOpt[String]
          val wordTargets = (jobject \ "targets").children.map(_.extract[String]).toArray

          (classifier, wordSource) match {
            case (Some(classname), Some(textA)) => functions :+= TextCosineOrder(classname, textA, wordTargets)
            case (_, _) => {}
          }
        }
        else if (!IntrinsicEvaluationTypes.cosineAnalogy.r.findAllMatchIn(label).isEmpty) {
          val classifier = (jobject \ "classifier").extract[String]
          val tests = (jobject \ "text").children.map(_.extract[String]).toArray.map(analogyString => {
            val Array(src, dst) = analogyString.split("\\s?\\=\\s?")
            val Array(wordA, wordB) = src.split("\\s?\\+\\s?")
            val Array(wordX, wordY) = dst.split("\\s?\\+\\s?")

            functions :+= Text3CosAddAnalogy(classifier+"@1", wordA, wordB, wordX, wordY)
            functions :+= Text3CosAddAnalogy(classifier+"@10", wordA, wordB, wordX, wordY, top=10)
            functions :+= Text3CosAddAnalogy(classifier+"@20", wordA, wordB, wordX, wordY, top=20)
            functions :+= Text3CosMulAnalogy(classifier+"@1", wordA, wordB, wordX, wordY)
            functions :+= Text3CosMulAnalogy(classifier+"@10", wordA, wordB, wordX, wordY, top=10)
            functions :+= Text3CosMulAnalogy(classifier+"@20", wordA, wordB, wordX, wordY, top=20)

          })

          val filename = (jobject \ "filename").extractOpt[String]
          filename match {
            case Some(fname) => {
              functions ++= Source.fromFile(fname, "UTF-8").getLines().map(_.trim)
                .map(line => line.split("\\s+")).flatMap(test =>
                  Array(Text3CosAddAnalogy(classifier, test(0), test(1), test(2), test(3)),
                    Text3CosMulAnalogy(classifier, test(0), test(1), test(2), test(3))))
            }
            case None => {}
          }
        }

      }
    }

  }

}

object IntrinsicEvaluationTypes {
  val cosineSimilarity = "(cosine\\-similarity|similarity)"
  val cosineAnalogy = "(cosine\\-add\\-analogy|analogy)"
  val morphAnalogy = "(morph\\-analogy|morphology)"
  val clusterSimilarity = "(cluster\\-similarity|cluster)"
  val classifierSimilarity = "(classification\\-similarity|classification)"
}


object IntrinsicEvaluation{

  def mergeText(): Unit = {
    val fnames = Array("sentences-tr.txt","sentences-may-v1-tr.txt","tud.txt")
    val outputFilename = "resources/text/may-join-tr.txt"
    val pw = new PrintWriter(outputFilename)

    fnames.foreach(fname=>{
      val filename = "resources/text/"+fname;
      Source.fromFile(filename).getLines().foreach(line => {
        pw.println(line)
      })

    })
    pw.close()
  }

  def extractTUD(): Unit = {

    val folder = "resources/dictionary/tud/"
    val sentenceFilename = "resources/text/tud.txt"
    val files = new File(folder).listFiles()
    val pw = new PrintWriter(sentenceFilename)
    files.foreach(file=>{
      Source.fromFile(file).getLines().drop(1).foreach(line=>{
        val split = line.split("\",\"").last
        val sentence = split.substring(0, split.length - 1)
        pw.println(sentence)
      })
    })

    pw.close()
  }

  def extractDictionary(): Unit = {
    val evaluationSet = "resources/text/evaluation-words.txt"
    val mainEvaluation = new IntrinsicEvaluation("extract")
      .attachEvaluations("resources/evaluations/sentence-tr.json")
      .compile()

    val words = mainEvaluation.universe()
    val pw = new PrintWriter(evaluationSet)

    words.foreach(word=>{
      pw.println(word)
    })

    pw.close()
  }

  def main(args: Array[String]): Unit = {
    //extractTUD()
    mergeText()
  }
}

