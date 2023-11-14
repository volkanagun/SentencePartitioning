package evaluation

import models.EmbeddingModel
import sampling.experiments.SampleParams
import utils.Params

import java.io.PrintWriter
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

/**
 * @author Volkan Agun
 */
class InstrinsicEvaluationReport extends Serializable {

  var truepositives = 0d
  var similarity = 0d
  var testpairCount = 0d
  var testoovCount = 0d
  var trainTime = 0d

  var classifierScoreMap = Map[String, Double]()
  var classifierSimilarityMap = Map[String, Double]()
  var classifierQueryCount = Map[String, Double]()
  var classifierOOVMap = Map[String, Double]()
  var classifierSkipCount = Map[String, Double]()

  def setTrainTime(totalTime: Double): this.type = {
    trainTime = totalTime / (1000 * 60)
    this
  }

  def incrementTestPair(): this.type = {
    synchronized {
      testpairCount += 1
      this
    }
  }

  def incrementOOV(): this.type = {
    synchronized {
      testoovCount += 1
      this
    }
  }

  def incrementTruePositives(value: Double): this.type = {
    synchronized {
      truepositives += value
      this
    }
  }

  def incrementSimilarity(value: Double): this.type = {
    synchronized {
      similarity += value
      this
    }
  }

  def update(map: Map[String, Double], key: String, value: Double): Map[String, Double] = {
    synchronized {
      val score = map.getOrElse(key, 0d) + value
      val result = map.updated(key, score)
      result
    }

  }

  def incrementScoreMap(key: String, value: Double): this.type = {
    synchronized {
      classifierScoreMap = update(classifierScoreMap, key, value)
      this
    }
  }

  def incrementSimilarityMap(key: String, value: Double): this.type = {
    synchronized {
      classifierSimilarityMap = update(classifierSimilarityMap, key, value)
      this
    }
  }

  def incrementQueryCount(key: String, value: Double): this.type = {
    synchronized {
      classifierQueryCount = update(classifierQueryCount, key, value)
      this
    }
  }

  def incrementOOVCount(key: String, value: Double): this.type = {
    synchronized {
      classifierOOVMap = update(classifierOOVMap, key, value)
      this
    }
  }

  def incrementSkipCount(key: String, value: Double): this.type = {
    synchronized {
      classifierSkipCount = update(classifierSkipCount, key, value)
      this
    }

  }

  def append(report: InstrinsicEvaluationReport): this.type = {
    synchronized {
      truepositives += report.truepositives
      testpairCount += report.testpairCount
      similarity += report.similarity
      report.classifierScoreMap.foreach { case (key, value) => classifierScoreMap = classifierScoreMap.updated(key, value + classifierScoreMap.getOrElse(key, 0d)) }
      report.classifierQueryCount.foreach { case (key, value) => classifierQueryCount = classifierQueryCount.updated(key, value + classifierQueryCount.getOrElse(key, 0d)) }
      report.classifierOOVMap.foreach { case (key, value) => classifierOOVMap = classifierOOVMap.updated(key, value + classifierOOVMap.getOrElse(key, 0d)) }
      report.classifierSkipCount.foreach { case (key, value) => classifierSkipCount = classifierSkipCount.updated(key, value + classifierSkipCount.getOrElse(key, 0d)) }
      report.classifierSimilarityMap.toArray.foreach { case (key, value) => classifierSimilarityMap = classifierSimilarityMap.updated(key, value + classifierSimilarityMap.getOrElse(key, 0d)) }
      this
    }
  }

  override def toString: String = {
    var text = ""

    text += s"Test/Query Count: ${testpairCount}"
    text += s"True Positives: ${truepositives}"
    text += s"Similarity: ${similarity / testpairCount}"
    text += s"Classifier Scores: \n${classifierScoreMap.toArray.map { case (name, score) => s"Classifier: ${name} Score:${score}\n" }.mkString("\n")}"
    text += s"Classifier Similarity: \n${classifierSimilarityMap.toArray.map { case (name, score) => s"Classifier: ${name} Score:${score}\n" }.mkString("\n")}"
    text += s"Classifier Test/Query Counts: \n${classifierQueryCount.toArray.map { case (name, score) => s"Classifier: ${name}  Test/Query Count:${score}\n" }.mkString("\n")}"
    text += s"Classifier Out of Vocabulary Counts: \n${classifierOOVMap.toArray.map { case (name, count) => s"Classifier: ${name} OOV Count:${count}\n" }.mkString("\n")}"
    text += s"Classifier Skipped Counts: \n${classifierSkipCount.toArray.map { case (name, count) => s"Classifier: ${name}" }}"
    val skippedClassifiers = classifierSkipCount.map { case (name, count) => (name, classifierQueryCount(name) == count) }
      .filter(_._2)

    text += s"Skipped Classifiers: \n${skippedClassifiers.toArray.map { case (name, _) => "Classifier: " + name }.mkString("\n")}"

    text

  }

  def print(): Unit = {
    println(s"Test/Query Count: ${testpairCount}")
    println(s"True Positives: ${truepositives}")
    println(s"Similarity: ${similarity / testpairCount}")
    println(s"Score: ${similarity / testpairCount}")
    println(s"Training Time: ${trainTime}")

    val skippedClassifiers = classifierSkipCount.map { case (name, count) => (name, classifierQueryCount(name) == count) }
      .filter(_._2)
    val scoreText = classifierScoreMap.toArray.map { case (name, score) => {
      "Classifier: " + name + "| Score: " + (100 * score + Double.MinPositiveValue) / (classifierQueryCount(name) + Double.MinPositiveValue)
    }
    }.mkString("\n\t")

    val similarityText = classifierSimilarityMap.toArray.map { case (name, score) => {
      "Classifier: " + name + "| Similarity: " + (100 * score + Double.MinPositiveValue) / (classifierSimilarityMap(name) + Double.MinPositiveValue)
    }
    }.mkString("\n\t")

    val countText = classifierQueryCount.toArray.map { case (name, count) => {
      "Classifier: " + name + "| Test/Query Count: " + count
    }
    }.mkString("\n\t")
    val oovText = classifierOOVMap.toArray.map { case (name, count) => {
      "Classifier: " + name + "| OOV Count: " + count
    }
    }.mkString("\n\t")
    val skipText = classifierSkipCount.toArray.map { case (name, count) => {
      "Classifier: " + name + "| Skip Count: " + count
    }
    }.mkString("\n\t")
    val skipClassifiersText = skippedClassifiers.toArray.map { case (name, _) => "Classifier: " + name }.mkString("\n\t")
    println(s"Classifier Scores: \n${scoreText}")
    println(s"Classifier Similarity: \n${similarityText}")
    println(s"Classifier Test/Query Counts: \n${countText}")
    println(s"Classifier Out of Vocabulary Counts: \n${oovText}")
    println(s"Classifier Skipped Counts: \n${skipText}")
    println(s"Skipped Classifiers: \n${skipClassifiersText}")
  }

  def printByGroup(pw: PrintWriter, params: String, groups: Array[String]): Unit = {

    pw.println(params)
    pw.println(s"Total Test/Query Count: ${testpairCount}")
    pw.println(s"Total OOV Count: ${testpairCount}")
    pw.println(s"True Positives: ${truepositives}")
    pw.println(s"Similarity: ${similarity / testpairCount}")


    val skippedClassifiers = classifierSkipCount.map { case (name, count) => (name, classifierQueryCount(name) == count) }
      .filter(_._2)
    val scoreText = classifierScoreMap.toArray.map { case (name, score) => {
      "Classifier: " + name + "| Score: " + (100 * score + Double.MinPositiveValue) / (classifierScoreMap(name) + Double.MinPositiveValue)
    }
    }.mkString("\n\t")

    val countText = classifierQueryCount.toArray.map { case (name, score) => {
      "Classifier: " + name + "| Test/Query Count: " + score
    }
    }.mkString("\n\t")
    val oovText = classifierOOVMap.toArray.map { case (name, count) => {
      "Classifier: " + name + "| OOV Count: " + count
    }
    }.mkString("\n\t")
    val skipText = classifierSkipCount.toArray.map { case (name, count) => {
      "Classifier: " + name + "| Skip Count: " + count
    }
    }.mkString("\n\t")

    val similarityText = classifierSimilarityMap.toArray.map { case (name, count) => {
      "Classifier: " + name + "| Similarity: " + count
    }
    }.mkString("\n\t")

    val skipClassifiersText = skippedClassifiers.toArray.map { case (name, _) => "Skipped Classifier: " + name }.mkString("\n\t")
    pw.println(s"Classifier Scores: \n${scoreText}")
    pw.println(s"Classifier Similarity: \n${similarityText}")
    pw.println(s"Classifier Test/Query Counts: \n${countText}")
    pw.println(s"Classifier Out of Vocabulary Counts: \n${oovText}")
    pw.println(s"Classifier Skipped Counts: \n${skipText}")
    pw.println(s"Skipped Classifiers: \n${skipClassifiersText}")

    groups.foreach(group => {
      val lwgroupname = group.toLowerCase
      pw.println(s"Statistics for ${group}")
      pw.println(s"+++++++++++++++++++++++++++++++++++++++++++++++++++")
      val groupSkips = classifierSkipCount.map { case (name, count) => (name, classifierQueryCount(name) == count) }
        .filter(_._1.toLowerCase().contains(lwgroupname)).toArray
        .map { case (name, _) => {
          "Skipped Classifier: " + name
        }
        }.mkString("\n\t")
      val groupCountArray = classifierQueryCount.filter(_._1.toLowerCase.contains(lwgroupname))
      val groupCountSum = groupCountArray.toArray.map(_._2).sum
      val groupCount = "Group: " + group + "| Count: " + groupCountSum + "| Size: " + groupCountArray.size + "\n\t"
      val groupScores = classifierScoreMap.filter(_._1.toLowerCase.contains(lwgroupname))

      val groupSimilarity = classifierSimilarityMap.filter(_._1.toLowerCase.contains(lwgroupname))
      val groupSimilaritySum = groupSimilarity.toArray.map(_._2).sum / groupSimilarity.size
      val groupSimilarityScore = "Group: " + group + "| Similarity Score: " + groupSimilaritySum + "| Similarity Count: " + groupSimilarity.size

      val groupScoresSum = groupScores.toArray.map(_._2).sum / groupCountSum
      val groupScore = "Group: " + group + "| Score: " + groupScoresSum + "| Size: " + groupScores.size + "\n\t"
      val groupOOVs = classifierOOVMap.filter(_._1.toLowerCase.contains(lwgroupname)).toArray.map(_._2)
      val groupOOVSum = groupOOVs.sum
      val groupOOV = "Group: " + group + "| OOV Count: " + groupOOVSum + "| Size: " + groupOOVs.size + "\n\t"
      val groupSkipArray = classifierSkipCount.filter(_._1.toLowerCase.contains(lwgroupname))
      val groupSkipSum = groupSkipArray.toArray.map(_._2).sum
      val groupSkipCounts = "Group: " + group + "| Skip Count: " + groupSkipSum + "| Count: " + groupSkipArray.size + "\n\t"

      pw.println(groupCount)
      pw.println(groupSimilarityScore)
      pw.println(groupSkips)
      pw.println(groupScore)
      pw.println(groupOOV)
      pw.println(groupSkipCounts)
    })

  }

  def printXMLTag(tag: String, content: String): String = {
    "<" + tag + ">" + content + "</" + tag + ">"
  }

  def printXMLTag(tag: String, attribute: String, name: String, value: String, score: Double): String = {
    "<" + tag + " " + attribute + "=\"" + name + "\" " + value + "=\"" + score + "\"/>"
  }

  def printXMLTag(tag: String, content: Double): String = {
    printXMLTag(tag, content.toString)
  }

  def printXML(pw: PrintWriter, params: String): Unit = {
    printXMLByGroup(pw, params, classifierQueryCount.keys.toArray)
  }

  def printXMLByGroup(pw: PrintWriter, params: String, groups: Array[String]): Unit = {

    pw.println("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    pw.println("<ROOT>")
    pw.println(params)
    pw.println(printXMLTag("QUERY_COUNT", testpairCount))
    pw.println(printXMLTag("OOV_COUNT", testoovCount))
    pw.println(printXMLTag("OOV_RATE", testoovCount / testpairCount))
    pw.println(printXMLTag("EFFICIENCY", trainTime))
    pw.println(printXMLTag("TRUE_COUNT", truepositives))
    pw.println(printXMLTag("TRUE_RATE", truepositives / testpairCount))
    pw.println(printXMLTag("SIMILARITY", similarity / testpairCount));
    pw.println(printXMLTag("F1-MEASURE", similarity / testpairCount));
    pw.println(printXMLTag("ACCURACY", truepositives / testpairCount));


    val skippedClassifiers = classifierSkipCount.map { case (name, count) => (name, classifierQueryCount(name) == count) }
      .filter(_._2)

    val scoreText = classifierScoreMap.toArray.map { case (name, score) => {
      printXMLTag("CLASSIFIER", "NAME", name, "SCORE", (100 * score) / (classifierScoreMap(name) + Double.MinPositiveValue))
    }
    }.mkString("\n")

    val countText = classifierQueryCount.toArray.map { case (name, score) => {
      printXMLTag("CLASSIFIER", "NAME", name, "COUNT", score)
    }
    }.mkString("\n")

    val oovText = classifierOOVMap.toArray.map { case (name, count) => {
      printXMLTag("CLASSIFIER", "NAME", name, "OOV", count)
    }
    }.mkString("\n")
    val skipText = classifierSkipCount.toArray.map { case (name, count) => {
      printXMLTag("CLASSIFIER", "NAME", name, "SKIP", count)
    }
    }.mkString("\n")

    val similarityText = classifierSimilarityMap.toArray.map { case (name, count) => {
      val queryCount = classifierQueryCount(name)
      val sim = (count) / (queryCount + Double.MinPositiveValue)
      printXMLTag("CLASSIFIER", "NAME", name, "SIMILARITY", sim)
    }
    }.mkString("\n")

    val skipClassifiersText = skippedClassifiers.toArray.map { case (name, _) => "<SKIPPED NAME=\"" + name + "\"/>" }.mkString("\n")
    pw.println("<ACCURACY>")
    pw.println(scoreText)
    pw.println("</ACCURACY>")

    pw.println("<SIMILARITY>")
    pw.println(similarityText)
    pw.println("</SIMILARITY>")

    pw.println("<COUNTS>")
    pw.println(countText)
    pw.println("</COUNTS>")

    pw.println("<OOV>")
    pw.println(oovText)
    pw.println("</OOV>")

    pw.println("<SKIPPED_COUNT>")
    pw.println(skipText)
    pw.println("</SKIPPED_COUNT>")

    pw.println("<SKIPPED_CLASSIFIERS>")
    pw.println(skipClassifiersText)
    pw.println("</SKIPPED_CLASSIFIERS>")

    pw.println("<GROUPS>")
    groups.foreach(group => {

      val lwgroupname = group.toLowerCase
      pw.println(s"<GROUP NAME=\"${group}\">")
      val groupSkips = classifierSkipCount.map { case (name, count) => (name, classifierQueryCount(name) == count) }
        .filter(_._1.toLowerCase().contains(lwgroupname)).toArray
        .map { case (name, _) => {
          printXMLTag("SKIP", name)
        }
        }.mkString("\n")

      pw.println("<SKIPS>")
      pw.println(groupSkips)
      pw.println("</SKIPS>")

      val groupCountArray = classifierQueryCount.filter(_._1.toLowerCase.contains(lwgroupname))
      val groupCountSum = groupCountArray.toArray.map(_._2).sum


      val groupCount = printXMLTag("CLASSIFIER", "NAME", group, "COUNT", groupCountSum)
      val groupScores = classifierScoreMap.filter(_._1.toLowerCase.contains(lwgroupname))

      val groupSimilarity = classifierSimilarityMap.filter(_._1.toLowerCase.contains(lwgroupname))
      val groupSimilaritySum = (groupSimilarity.toArray.map(_._2).sum + Double.MinPositiveValue) / (groupCountSum + Double.MinPositiveValue)
      val groupSimilarityScore = printXMLTag("CLASSIFIER", "NAME", group, "SIMILARITY", groupSimilaritySum)

      val groupScoresSum = (groupScores.toArray.map(_._2).sum + Double.MinPositiveValue) / (groupCountSum + Double.MinPositiveValue)
      val groupScore = printXMLTag("CLASSIFIER", "NAME", group, "SCORE", groupScoresSum)
      val groupOOVs = classifierOOVMap.filter(_._1.toLowerCase.contains(lwgroupname)).toArray.map(_._2)
      val groupOOVSum = groupOOVs.sum
      val groupOOV = printXMLTag("CLASSIFIER", "NAME", group, "OOV", groupOOVSum)
      val groupSkipArray = classifierSkipCount.filter(_._1.toLowerCase.contains(lwgroupname))
      val groupSkipSum = groupSkipArray.toArray.map(_._2).sum
      val groupSkipCounts = printXMLTag("CLASSIFIER", "NAME", group, "SKIP", groupSkipSum)

      pw.println("<SCORES>")
      pw.println(groupCount)
      pw.println(groupSimilarityScore)
      pw.println(groupScore)
      pw.println(groupOOV)
      pw.println(groupSkipCounts)
      pw.println("</SCORES>")
      pw.println("</GROUP>")
    })

    pw.println("</GROUPS>")
    pw.println("</ROOT>")
  }

  def printProgress(classifier: String): Unit = {
    println(s"Evaluating ${classifier}")
  }
}

case class EvalScore(tp: Double, similarity: Double)

abstract class IntrinsicFunction() extends Serializable {

  def evaluate(model: EmbeddingModel): EvalScore

  def setDictionary(set: Set[String], model: EmbeddingModel): this.type

  def setWords(set: Set[String]): this.type

  def setEmbeddings(set: Set[(String, Array[Float])]): this.type

  def count(): Int

  def universe(): Set[String]

  def getClassifier(): String

  def filter(group: Array[String]): Boolean

  def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport

  def contains(dictionary: Map[String, Array[Float]], words: Array[String]): Boolean = {
    words.par.exists(subword => dictionary.contains(subword))
  }

  def embeddings(model: EmbeddingModel, item: String): Array[Float] = {
    val itemArray = model.forward(item)
    itemArray
  }

  def embeddings(model: EmbeddingModel, items: Array[String]): Array[Float] = {
    val itemArray = items.map(item => scalar(model, item))
    avg(itemArray)
  }

  def cosineDistance(source: Array[Float], destination: Array[Float]): Float = {
    val nom = source.zip(destination).map(pair => pair._1 * pair._2).sum + Double.MinPositiveValue
    val denom = Math.sqrt(source.map(item => item * item).sum) * Math.sqrt(destination.map(item => item * item).sum) + Double.MinPositiveValue
    val result = 1 - (nom / denom)
    result.toFloat
  }

  def angularSimilarity(source: Array[Float], destination: Array[Float]): Float = {
    val nom = source.zip(destination).map(pair => pair._1 * pair._2).sum + Double.MinPositiveValue
    val denom = Math.sqrt(source.map(item => item * item).sum) * Math.sqrt(destination.map(item => item * item).sum) + Double.MinPositiveValue
    val score = nom / denom
    val result = (1 - math.acos(score) / math.Pi).toFloat
    if (result.isNaN) {
      0f
    }
    else {
      result
    }
  }

  def difference(source: Array[Float], destination: Array[Float]): Array[Float] = {
    source.zip(destination).map(pair => pair._1 - pair._2).toArray
  }

  def add(source: Array[Float], destination: Array[Float]): Array[Float] = {
    source.zip(destination).map(pair => pair._1 + pair._2)
  }

  def div(source: Array[Float], scalar: Float): Array[Float] = {
    source.map(pair => pair / scalar)
  }

  def cosine(model: EmbeddingModel, source: String, destination: String): Float = {
    val tensorSource = embeddings(model, source)
    val tensorDestination = embeddings(model, destination)
    cosineDistance(tensorSource, tensorDestination)
  }

  def cosineAngular(model: EmbeddingModel, source: String, destination: String): Float = {
    val tensorSource = embeddings(model, source)
    val tensorDestination = embeddings(model, destination)
    angularSimilarity(tensorSource, tensorDestination)
  }

  def cosine(model: EmbeddingModel, source: Array[String], destination: Array[String]): Float = {
    val tensorSource = embeddings(model, source)
    val tensorDestination = embeddings(model, destination)
    cosineDistance(tensorSource, tensorDestination)
  }

  def cosineAngular(model: EmbeddingModel, source: Array[String], destination: Array[String]): Float = {
    val tensorSource = embeddings(model, source)
    val tensorDestination = embeddings(model, destination)
    angularSimilarity(tensorSource, tensorDestination)
  }

  def cosineAngular(model: EmbeddingModel, source: Array[Float], destination: Array[String]): Float = {
    val tensorSource = source
    val tensorDestination = embeddings(model, destination)
    angularSimilarity(tensorSource, tensorDestination)
  }

  def cosineAngular(model: EmbeddingModel, tensorSource: Array[Float], destination: String): Float = {
    val tensorDestination = embeddings(model, destination)
    angularSimilarity(tensorSource, tensorDestination)
  }

  def cosineAngular(tensorSource: Array[Float], tensorDestination: Array[Float]): Float = {
    angularSimilarity(tensorSource, tensorDestination)
  }

  def cosine(model: EmbeddingModel, tensorSource: Array[Float], destination: String): Float = {
    val tensorDestination = embeddings(model, destination)
    cosineDistance(tensorSource, tensorDestination)
  }

  def cosine(tensorSource: Array[Float], tensorDestination: Array[Float]): Float = {
    cosineDistance(tensorSource, tensorDestination)
  }


  def analogy(model: EmbeddingModel, wordA: String, wordB: String, wordC: String): Array[Float] = {
    val tensorA = embeddings(model, wordA)
    val tensorB = embeddings(model, wordB)
    val tensorC = embeddings(model, wordC)
    add(tensorC, difference(tensorA, tensorB))
  }

  def analogy(model: EmbeddingModel, wordA: Array[String], wordB: Array[String], wordC: Array[String]): Array[Float] = {
    val tensorA = embeddings(model, wordA)
    val tensorB = embeddings(model, wordB)
    val tensorC = embeddings(model, wordC)
    add(tensorC, difference(tensorA, tensorB))
  }


  def analogyAvg(model: EmbeddingModel, pairs: Array[(String, String)], wordY: String): Array[Float] = {
    val tensors = pairs.map { case (s, d) => difference(embeddings(model, s), embeddings(model, d)) }
    val size = tensors.length
    val wordYTensor = embeddings(model, wordY)
    val sumTensor = tensors.foldLeft[Array[Float]](Array.fill[Float](size)(0f)) { case (main, tensor) => add(main, tensor) }
    val diffAvgTensor = div(sumTensor, size)
    add(wordYTensor, diffAvgTensor)
  }

  def analogyAvgArray(model: EmbeddingModel, pairs: Array[(Array[String], Array[String])], wordY: Array[String]): Array[Float] = {
    val tensors = pairs.map { case (s, d) => difference(embeddings(model, s), embeddings(model, d)) }
    val size = tensors.length
    val wordYTensor = embeddings(model, wordY)
    val sumTensor = tensors.foldLeft[Array[Float]](Array.fill[Float](size)(0f)) { case (main, tensor) => add(main, tensor) }
    val diffAvgTensor = div(sumTensor, size)
    add(wordYTensor, diffAvgTensor)
  }

  def avg(tensors: Array[Array[Float]]): Array[Float] = {
    val size = tensors.head.length
    val sumTensor = tensors.foldLeft[Array[Float]](Array.fill[Float](size)(0f)) { case (main, tensor) => add(main, tensor) }
    div(sumTensor, tensors.length)
  }

  def scalar(model: EmbeddingModel, wv: String): Array[Float] = {
    val array = model.forward(wv)
    array.map(f => f * wv.length)
  }
}

case class TextCosineOrder(var classifier: String, val wordSource: String, val wordTargets: Array[String], top: Int = 1) extends IntrinsicFunction {
  classifier = "|SimOrder| - " + classifier

  var tokenizer: (String => Array[String]) = (item: String) => Array(item)

  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = this


  override def setWords(set: Set[String]): TextCosineOrder.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): TextCosineOrder.this.type = this

  override def getClassifier(): String = classifier


  override def filter(group: Array[String]): Boolean = group.forall(item => classifier.contains(item))

  def setTokenizer(tokenizer: (String => Array[String])): this.type = {
    this.tokenizer = tokenizer
    this
  }

  override def count(): Int = 1

  override def universe(): Set[String] = Set(wordSource) ++ wordTargets.toSet

  override def evaluate(model: EmbeddingModel): EvalScore = {
    val wordSourceSet = tokenizer(wordSource)

    val similarityOrder = wordTargets.map(target => {
      val targetSet = tokenizer(target)
      (target, cosineAngular(model, wordSourceSet, targetSet))
    }).sortBy(_._2).reverse

    val orders = similarityOrder.map(_._1)
    val angularScore = similarityOrder.map(_._2).sum / wordTargets.size

    val orderScore = orders.zipWithIndex.map { case (similarityTarget, index) => math.abs(wordTargets.indexOf(similarityTarget) - index) }.sum.toDouble / wordTargets.length
    EvalScore(orderScore, angularScore)
  }


  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport()
      .incrementTestPair()
    val dictionary = model.getDictionary()
    ier.incrementQueryCount(classifier, 1d)
    val wordAset = model.tokenize(wordSource)
    if (contains(dictionary, wordAset)) {
      val filteredTargets = wordTargets.filter(target => dictionary.contains(target))

      if (filteredTargets.isEmpty) {
        ier.incrementOOVCount(classifier, 1d)
        ier.incrementScoreMap(classifier, 0d)
        ier.incrementSkipCount(classifier, 1d)

      }
      else {
        val evals = evaluate(model)
        ier.incrementScoreMap(classifier, evals.tp)
        ier.incrementScoreMap(classifier, evals.similarity)
        ier.incrementSimilarity(evals.similarity)
      }


    }
    else {

      ier.incrementSkipCount(classifier, 1d)
      ier.incrementOOVCount(classifier, 1d)
    }
  }

}

case class Text3CosAddAnalogyOrder(var classifier: String, val wordA: String, val wordB: String, val wordX: String, val wordY: String, val testSet: Array[String], top: Int = 1) extends IntrinsicFunction {
  classifier = "|3CosAddOrder| - " + classifier
  require(testSet.contains(wordY), s"${wordY} must be included in testSet")

  var tokenizer: (String) => Array[String] = null;
  var skip = false

  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = {
    this
  }


  override def filter(group: Array[String]): Boolean = group.forall(item => classifier.contains(item))

  override def setWords(set: Set[String]): Text3CosAddAnalogyOrder.this.type = this

  override def setEmbeddings(set: Set[(String, Array[Float])]): Text3CosAddAnalogyOrder.this.type = this

  override def getClassifier(): String = classifier

  override def count(): Int = 1


  override def universe(): Set[String] = Set(wordA, wordB, wordX, wordY) ++ testSet.toSet

  def evaluateSingle(model: EmbeddingModel): EvalScore = {
    val dictionary = model.getDictionary()
    val wordAset = tokenizer(wordA)
    val wordBset = tokenizer(wordB)
    val wordXset = tokenizer(wordX)
    val wordYset = tokenizer(wordY)

    if (contains(dictionary, wordAset) && contains(dictionary, wordBset) && contains(dictionary, wordXset) && contains(dictionary, wordYset)) {

      val analogyTensor = analogy(model, wordAset, wordBset, wordXset)
      val score = cosineAngular(model, analogyTensor, wordYset)
      val foundItems = testSet.par.exists(item => cosineAngular(model, analogyTensor, item) > score)
      val tp = if (!foundItems) 1.0 else 0.0
      EvalScore(tp, score)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }

  override def evaluate(model: EmbeddingModel): EvalScore = {
    if (top == 1) evaluateSingle(model)
    else evaluateAll(model)
  }


  def evaluateAll(model: EmbeddingModel): EvalScore = {
    val analogyTensor = analogy(model, wordA, wordB, wordX)
    val score = cosineAngular(model, analogyTensor, wordY)
    val foundItems = testSet.map(item => (item, cosineAngular(model, analogyTensor, item))).sortBy(_._2).reverse.take(top).map(_._1)
    val tp = if (foundItems.contains(wordY)) 1.0 else 0.0
    EvalScore(tp, score)
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport().incrementTestPair()
    ier.incrementQueryCount(classifier, 1d)
    val evalScore = evaluate(model)


    if (skip) {
      ier.incrementOOVCount(classifier, 1d)
      ier.incrementSkipCount(classifier, 1d)
      ier.incrementOOV()
    }
    else {
      ier.incrementTruePositives(evalScore.tp)
      ier.incrementScoreMap(classifier, evalScore.tp)

      ier.incrementSimilarityMap(classifier, evalScore.similarity)
      ier.incrementSimilarity(evalScore.similarity)
    }
  }

}

abstract class TextAnalogy(var skip: Boolean = false) extends IntrinsicFunction {
  var testWords = Set[String]()
  var testEmbeddings = Set[(String, Array[Float])]()
}

case class Text3CosAddAnalogy(var classifier: String, val wordA: String, val wordB: String, val wordX: String, val wordY: String, skipx: Boolean = false, top: Int = 1) extends TextAnalogy(skipx) {
  classifier = "|3CosAdd| - " + classifier

  var testSources = Set(wordA, wordB, wordX)

  override def filter(group: Array[String]): Boolean = {
    group.forall(item => classifier.contains(item))
  }

  override def getClassifier(): String = classifier

  override def count(): Int = 1

  override def universe(): Set[String] = Set(wordA, wordB, wordX, wordY)


  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = {
    this
  }


  override def setWords(set: Set[String]): Text3CosAddAnalogy.this.type = {
    testWords = set.filter(target => !testSources.contains(target))
    this
  }

  override def setEmbeddings(set: Set[(String, Array[Float])]): Text3CosAddAnalogy.this.type = {
    testEmbeddings = set.filter(pair => !testSources.contains(pair._1))
    this
  }

  def evaluateSingle(model: EmbeddingModel): EvalScore = {

    val dictionary = model.getDictionary()
    val wordAset = model.tokenize(wordA)
    val wordBset = model.tokenize(wordB)
    val wordXset = model.tokenize(wordX)
    val wordYset = model.tokenize(wordY)

    if (contains(dictionary, wordAset) && contains(dictionary, wordBset) && contains(dictionary, wordXset) && contains(dictionary, wordYset)) {
      val analogyTensor = analogy(model, wordAset, wordBset, wordXset)
      val wordYavg = avg(wordYset.map(wy => scalar(model, wy)))
      val similarityScore = cosineAngular(analogyTensor, wordYavg)
      val foundItem = testEmbeddings.toArray.par.exists { pair => cosineAngular(analogyTensor, pair._2) > similarityScore }
      val tp = if (!foundItem) 1.0 else 0.0
      EvalScore(tp, similarityScore)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }

  def evaluateAll(model: EmbeddingModel): EvalScore = {

    val dictionary = model.getDictionary()
    val wordAset = model.tokenize(wordA)
    val wordBset = model.tokenize(wordB)
    val wordXset = model.tokenize(wordX)
    val wordYset = model.tokenize(wordY)

    if (contains(dictionary, wordAset) && contains(dictionary, wordBset) && contains(dictionary, wordXset) && contains(dictionary, wordYset)) {
      val analogyTensor = analogy(model, wordAset, wordBset, wordXset)
      val similarity = cosineAngular(analogyTensor, embeddings(model, wordYset))
      val foundItem = testEmbeddings.toArray.par.map { pair => (pair._1, cosineAngular(analogyTensor, pair._2)) }
        .toArray
        .sortBy(_._2)
        .reverse
        .map(_._1)
        .take(top)

      val tp = if (wordYset.exists(item => foundItem.contains(item))) 1.0 else 0.0
      EvalScore(tp, similarity)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }


  override def evaluate(model: EmbeddingModel): EvalScore = {
    if (top == 1) evaluateSingle(model)
    else evaluateAll(model)
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport().incrementTestPair()
    ier.incrementQueryCount(classifier, 1d)

    val value = evaluate(model)

    if (skip) {
      ier.incrementSkipCount(classifier, 1d)
      ier.incrementOOVCount(classifier, 1d)
      ier.incrementOOV()
    }
    else {
      ier.incrementTruePositives(value.tp)
      ier.incrementScoreMap(classifier, value.tp)

      ier.incrementSimilarity(value.similarity)
      ier.incrementSimilarityMap(classifier, value.similarity)
    }


    ier.printProgress(classifier)
    ier
  }

}

case class Text3CosMulAnalogy(var classifier: String, val wordA: String, val wordB: String, val wordX: String, val wordY: String, skipx: Boolean = false, top: Int = 1) extends TextAnalogy(skipx) {
  classifier = "|3CosMul| - " + classifier

  var sources = Set(wordA, wordB, wordX, wordY)

  override def getClassifier(): String = classifier


  override def filter(group: Array[String]): Boolean = group.forall(item => classifier.contains(item))

  override def count(): Int = 1

  override def universe(): Set[String] = Set(wordA, wordB, wordX, wordY)


  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = {
    this
  }


  override def setWords(set: Set[String]): Text3CosMulAnalogy.this.type = {
    testWords = set.filter(target => !sources.contains(target))
    this
  }

  override def setEmbeddings(set: Set[(String, Array[Float])]): Text3CosMulAnalogy.this.type = {
    testEmbeddings = set.filter(pair => !sources.contains(pair._1))
    this
  }

  def evaluateSingle(model: EmbeddingModel): EvalScore = {

    val dictionary = model.getDictionary()
    val wordAset = model.tokenize(wordA)
    val wordBset = model.tokenize(wordB)
    val wordXset = model.tokenize(wordX)
    val wordYset = model.tokenize(wordY)

    if (contains(dictionary, wordAset) && contains(dictionary, wordBset) && contains(dictionary, wordXset) && contains(dictionary, wordYset)) {
      val wordAEmbeddings = embeddings(model, wordAset)
      val wordBEmbeddings = embeddings(model, wordBset)
      val wordXEmbeddings = embeddings(model, wordXset)
      val wordYEmbeddings = embeddings(model, wordYset)

      val n1 = cosine(wordBEmbeddings, wordYEmbeddings)
      val n2 = cosine(wordXEmbeddings, wordYEmbeddings)
      val n3 = cosine(wordYEmbeddings, wordAEmbeddings)

      val similarityScore = n1 * n2 / (n3 + 1E-10)
      val foundItem = testEmbeddings.toArray.par.exists { pair => {
        val n1T = cosine(wordBEmbeddings, pair._2)
        val n2T = cosine(wordXEmbeddings, pair._2)
        val n3T = cosine(pair._2, wordAEmbeddings)
        n1T * n2T / (n3T + 1E-10) > similarityScore
      }
      }
      val tp = if (!foundItem) 1.0 else 0.0
      EvalScore(tp, similarityScore)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }

  def evaluateAll(model: EmbeddingModel): EvalScore = {

    val dictionary = model.getDictionary()
    val wordAset = model.tokenize(wordA)
    val wordBset = model.tokenize(wordB)
    val wordXset = model.tokenize(wordX)
    val wordYset = model.tokenize(wordY)

    if (contains(dictionary, wordAset) && contains(dictionary, wordBset) && contains(dictionary, wordXset) && contains(dictionary, wordYset)) {

      val wordAEmbeddings = embeddings(model, wordAset)
      val wordBEmbeddings = embeddings(model, wordBset)
      val wordXEmbeddings = embeddings(model, wordXset)
      val wordYEmbeddings = embeddings(model, wordYset)

      val n1 = cosine(wordBEmbeddings, wordYEmbeddings)
      val n2 = cosine(wordXEmbeddings, wordYEmbeddings)
      val n3 = cosine(wordYEmbeddings, wordAEmbeddings)

      val similarityScore = n1 * n2 / (n3 + 1E-10)
      val foundItem = testEmbeddings.toArray.par.map { pair => {
        val n1T = cosine(wordBEmbeddings, pair._2)
        val n2T = cosine(wordXEmbeddings, pair._2)
        val n3T = cosine(pair._2, wordAEmbeddings)
        (pair._1, n1T * n2T / (n3T + 1E-10))
      }
      }.toArray.sortBy(_._2).reverse.map(_._1).take(top)

      val tp = if (wordYset.exists(item => foundItem.contains(item))) 1.0 else 0.0
      EvalScore(tp, similarityScore)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }


  override def evaluate(model: EmbeddingModel): EvalScore = {
    if (top == 1) evaluateSingle(model)
    else evaluateAll(model)
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport().incrementTestPair()
    ier.incrementQueryCount(classifier, 1d)

    val value = evaluate(model)

    if (skip) {
      ier.incrementSkipCount(classifier, 1d)
      ier.incrementOOVCount(classifier, 1d)
      ier.incrementOOV()
    }
    else {
      ier.incrementTruePositives(value.tp)
      ier.incrementScoreMap(classifier, value.tp)

      ier.incrementSimilarity(value.similarity)
      ier.incrementSimilarityMap(classifier, value.similarity)
    }


    ier.printProgress(classifier)
    ier
  }

}

case class Text3CosAvgAnalogy(var classifier: String, sourcePairs: Array[(String, String)], wordX: String, wordY: String, skipx: Boolean = false, top: Int = 1) extends TextAnalogy(skipx) {

  classifier = "|3CosAvg| - " + classifier

  var excludeSet = Set(wordX, wordY)
  var tokenizer: (String => Array[String]) = (item: String) => Array(item)

  def setTokenizer(tokenizer: (String => Array[String])): this.type = {
    this.tokenizer = tokenizer
    this
  }


  override def setWords(set: Set[String]): Text3CosAvgAnalogy.this.type = {
    testWords = set.filter(item => !excludeSet.contains(item))
    this
  }

  override def setEmbeddings(set: Set[(String, Array[Float])]): Text3CosAvgAnalogy.this.type = {
    testEmbeddings = set.filter(pair => !excludeSet.contains(pair._1))
    this
  }

  override def setDictionary(set: Set[String], model: EmbeddingModel): this.type = {

    this
  }

  override def getClassifier(): String = classifier


  override def filter(group: Array[String]): Boolean = group.forall(item => classifier.contains(item))

  override def count(): Int = 1


  override def universe(): Set[String] = {

    sourcePairs.flatMap(pair => Array(pair._1, pair._2)).toSet ++ Set(wordX, wordY)
  }

  def evaluateSingle(model: EmbeddingModel): EvalScore = {
    val map = model.getDictionary()
    val avaliablePairs = sourcePairs.map(pair => (model.tokenize(pair._1), model.tokenize(pair._2)))
      .map { case (r1, r2) => (r1.filter(p => contains(map, Array(p))), r2.filter(p => contains(map, Array(p)))) }
      .filter(pair => pair._1.nonEmpty && pair._2.nonEmpty)
    val wordXset = model.tokenize(wordX)
    val wordYset = model.tokenize(wordY)

    if (contains(map, wordXset) && contains(map, wordYset) && !avaliablePairs.isEmpty) {
      val analogyTensor = analogyAvgArray(model, avaliablePairs, wordXset)
      val similarity = cosineAngular(analogyTensor, embeddings(model, wordYset))
      val foundItems = testEmbeddings.exists { case (target, embedding) => {
        cosineAngular(analogyTensor, embedding) > similarity
      }
      }
      val tp = if (!foundItems) 1.0 else 0.0
      EvalScore(tp, similarity)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }

  def evaluateAll(model: EmbeddingModel): EvalScore = {
    val map = model.getDictionary()
    val avaliablePairs = sourcePairs.filter { case (r1, r2) => map.contains(r1) && map.contains(r2) }
    if (map.contains(wordX) && map.contains(wordY) && !avaliablePairs.isEmpty) {
      val analogyTensor = analogyAvg(model, avaliablePairs, wordX)
      val similarity = cosineAngular(analogyTensor, embeddings(model, wordY))
      val foundItems = testEmbeddings.toArray.map { case (target, array) => (target, cosineAngular(analogyTensor, array)) }
        .sortBy(_._2)
        .map(_._1)
        .reverse
        .take(top)

      val tp = (if (foundItems.contains(wordY)) 1.0 else 0.0)
      EvalScore(tp, similarity)
    }
    else {
      skip = true
      EvalScore(0d, 0d)
    }
  }

  override def evaluate(model: EmbeddingModel): EvalScore = {
    if (top == 1) evaluateSingle(model)
    else evaluateAll(model)
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport().incrementTestPair()

    val map = model.getDictionary()
    val avaliablePairs = sourcePairs.filter { case (r1, r2) => map.contains(r1) && map.contains(r2) }

    ier.incrementQueryCount(classifier, 1d)

    if (map.contains(wordX) && map.contains(wordY) && !avaliablePairs.isEmpty) {
      val score = evaluate(model)
      ier.incrementTruePositives(score.tp)
      ier.incrementScoreMap(classifier, score.tp)
      ier.incrementSimilarity(score.similarity)
      ier.incrementSimilarityMap(classifier, score.similarity)

    }
    else {
      ier.incrementOOVCount(classifier, 1d)
      ier.incrementSkipCount(classifier, 1d)
      ier.incrementOOV()
    }

    ier.printProgress(classifier)
    ier
  }
}


case class SemEvalAnalogyOrder(val classifier: String, val semevalid: String, val wordPairs: Array[(String, String)]) extends IntrinsicFunction {

  val testSet = wordPairs.map(_._1) ++ wordPairs.map(_._2)
  var generatedAnalogies = wordPairs.par.flatMap(p1 => wordPairs.map(p2 => (p1, p2))).map { case ((t1word1, t1word2), (t2word1, t2word2)) => {
    Text3CosAddAnalogyOrder(classifier, t1word1, t1word2, t2word1, t2word2, testSet.distinct)
  }
  }.toArray


  override def setDictionary(set: Set[String], model: EmbeddingModel): SemEvalAnalogyOrder.this.type = {
    generatedAnalogies.foreach(ier => ier.setDictionary(set, model))
    this
  }


  override def setWords(set: Set[String]): SemEvalAnalogyOrder.this.type = {
    generatedAnalogies.foreach(ier => ier.setWords(set))
    this
  }


  override def setEmbeddings(set: Set[(String, Array[Float])]): SemEvalAnalogyOrder.this.type = {
    generatedAnalogies.foreach(ier => ier.setEmbeddings(set))
    this
  }

  override def getClassifier(): String = classifier

  override def count(): Int = generatedAnalogies.map(_.count()).sum

  override def filter(group: Array[String]): Boolean = {
    generatedAnalogies = generatedAnalogies.filter(item => item.filter(group))
    generatedAnalogies.nonEmpty
  }

  override def universe(): Set[String] = {
    testSet.toSet
  }


  /*override def score(model: EmbedForward): Double = {
    generatedAnalogies.par.map(analogyTest => analogyTest.score(model)).toArray.sum / generatedAnalogies.length
  }*/

  override def evaluate(model: EmbeddingModel): EvalScore = {
    val scores = generatedAnalogies.par.map(analogyTest => analogyTest.evaluate(model))
    val tp = scores.map(_.tp)
      .toArray.sum / generatedAnalogies.length
    val similarity = scores.map(_.similarity).sum / generatedAnalogies.length

    EvalScore(tp, similarity)
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {
    val ier = new InstrinsicEvaluationReport()
    generatedAnalogies.par.map(analogy => analogy.evaluateReport(model, embedParams)).toArray.foreach(testQuery => {
      ier.append(testQuery)
    })

    ier
  }
}




case class SemEvalAnalogy(var classifier: String, semevalid: String, wordPairs: Array[(String, String)]) extends IntrinsicFunction {

  classifier = s"|SEMEVAL - ${semevalid}| - " + classifier
  var oovskip = false
  val generatedAddAnalogies = wordPairs.flatMap(p1 => wordPairs.map(p2 => (p1, p2)))
    .filter { case (pair1, pair2) => !pair1._1.equals(pair2._1) }
    .flatMap { case ((t1word1, t1word2), (t2word1, t2word2)) => {
      Array(
        Text3CosAddAnalogy(classifier + "@1", t1word1, t1word2, t2word1, t2word2, top = 1),
        Text3CosAddAnalogy(classifier + "@10", t1word1, t1word2, t2word1, t2word2, top = 10),
        Text3CosAddAnalogy(classifier + "@20", t1word1, t1word2, t2word1, t2word2, top = 20)
      )
    }
    }

  val generatedMulAnalogies = wordPairs.flatMap(p1 => wordPairs.map(p2 => (p1, p2)))
    .filter { case (pair1, pair2) => !pair1._1.equals(pair2._1) }
    .flatMap { case ((t1word1, t1word2), (t2word1, t2word2)) => {
      Array(
        Text3CosMulAnalogy(classifier + "@1", t1word1, t1word2, t2word1, t2word2, top = 1),
        Text3CosMulAnalogy(classifier + "@10", t1word1, t1word2, t2word1, t2word2, top = 10),
        Text3CosMulAnalogy(classifier + "@20", t1word1, t1word2, t2word1, t2word2, top = 20)
      )
    }
    }

  lazy val generatedAvgAnalogies = wordPairs.map(pair => Text3CosAvgAnalogy(classifier, wordPairs.filter(!_._1.equals(pair._1)), pair._1, pair._2))
  var allanalogies = generatedAddAnalogies ++ generatedMulAnalogies


  override def setDictionary(set: Set[String], model: EmbeddingModel): SemEvalAnalogy.this.type = {
    allanalogies.foreach(analogy => analogy.setDictionary(set, model))
    this
  }


  override def setWords(set: Set[String]): SemEvalAnalogy.this.type = {
    allanalogies.foreach(analogy => analogy.setWords(set))
    this
  }

  override def setEmbeddings(set: Set[(String, Array[Float])]): SemEvalAnalogy.this.type = {
    allanalogies.foreach(analogy => analogy.setEmbeddings(set))
    this
  }

  override def getClassifier(): String = classifier

  override def count(): Int = allanalogies.map(_.count()).sum


  override def filter(group: Array[String]): Boolean = {
    allanalogies = allanalogies.filter(analogy => analogy.filter(group))
    allanalogies.nonEmpty
  }

  override def universe(): Set[String] = {
    wordPairs.flatMap(pair => Array(pair._1, pair._2)).toSet
  }

  override def evaluate(model: EmbeddingModel): EvalScore = {
    val analogiesIV = allanalogies.par.map(analogyTest => (analogyTest, analogyTest.evaluate(model)))
      .toArray
      .filter(!_._1.skip).map(_._2)

    if (analogiesIV.length == 0) {
      oovskip = true
      EvalScore(0, 0d)
    }
    else {
      val similarity = analogiesIV.map(_.similarity).sum / analogiesIV.length
      val tp = analogiesIV.map(_.tp).sum / analogiesIV.length
      EvalScore(tp, similarity)
    }
  }

  override def evaluateReport(model: EmbeddingModel, embedParams: SampleParams): InstrinsicEvaluationReport = {

    val ier = new InstrinsicEvaluationReport()
    allanalogies.par.map(analogy => analogy.evaluateReport(model, embedParams)).toArray.foreach(testQuery => {
      ier.append(testQuery)
    })

    ier
  }

}

