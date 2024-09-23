package helper

import experiments.Params
import tagging.lemmatizer.{WordCollocation, WordLemmatizer}
import transducer.Transducer
import transducer.TransducerOp.locale

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.concurrent.TrieMap
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.io.Source

class LMDictionary {

  val binaryFilename = "resources/binary/ranking.bin"
  var dictionaryTextFilename = "resources/dictionary/lexicon.txt"
  val sentenceFilename = "resources/text/sentences/sentences-tr.txt"
  val split = "#"
  var dictionaryTrie = new LMNode(0)
  val alpha = 0.01

  def isEmpty():Boolean={
    dictionaryTrie.isEmpty()
  }

  def inference(token: String): Array[(String, Array[Double])] = {
    dictionaryTrie.decodeScores(token, split)
  }

  def filter(split: Array[(String, Array[Double])], topSplit: Int): Array[String] = {
    split.map{case(item, scores) => {
      val splitted = item.split("\\#")
       if(splitted.nonEmpty) {
         val score = splitted.zip(scores).map{case(suf, sc) => {
           val p = suf.length.toDouble / (item.length + alpha)
           sc * 1.0 / math.log(p)
         }}.sum / scores.length
         (item, -score)
       }
      else{
        (item, 0d)
      }
      }}.sortBy(_._2)
      .reverse
      .take(topSplit)
      .map(_._1)
  }

  def exists(): Boolean = {
    new File(binaryFilename).exists()
  }

  def normalize():this.type ={
    dictionaryTrie.normalize()
    this
  }

  def save(): this.type = {
    val objectStream = new ObjectOutputStream(new FileOutputStream(binaryFilename))
    dictionaryTrie.save(objectStream)
    objectStream.close()
    this
  }

  def load(): this.type = {
    println("Loading dictionary...")
    val objectStream = new ObjectInputStream(new FileInputStream(binaryFilename))
    dictionaryTrie.load(objectStream)
    objectStream.close()
    this
  }

  def fromDictionary(params: Params): LMDictionary = {

    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text lexicon...")
      Source.fromFile(dictionaryTextFilename).getLines()
        .map(line => line.split("\t").head.toLowerCase(locale)
          .replaceAll("[\\~\\&\\_]", "")
          .replaceAll("([au])([bcçdfgğhjklmnprsştvyz])E$", "$1$2a")
          .replaceAll("([eü])([bcçdfgğhjklmnprsştvyz])E$", "$1$2e").toLowerCase(locale).trim)
        .foreach(token => {
          dictionaryTrie.add(Array(token, "END"))
        })

      println("Constructing dictionary from text lexicon finished")
    }

    this
  }

  def fromText(params: Params): this.type = {
    val lemmatizer = WordLemmatizer()
    val maxSentence = params.lmMaxSentence
    println("Training dictionary from text")
    Range(0, params.lmEpocs).sliding(params.lmThreads, params.lmThreads).foreach(sequence => {
      sequence.toArray.par.map(index => {
        val start = index * maxSentence
        val node = new LMNode(0)
        Source.fromFile(sentenceFilename).getLines().zipWithIndex
          .filter(pair => pair._2 >= start).take(maxSentence).map(pair => pair._1).foreach(sentence => {
            val lemmaSplitList = lemmatizer.extract(sentence).map(wordGroup => {
              wordGroup.lemmaSplitList(split)
            })

            lemmaSplitList.foreach(lemmaSplitArray => {
              lemmaSplitArray.foreach(lemmaSplit => {
                node.add(lemmaSplit.split("\\#"))
              })
            })
          })

        node
      }).toArray.foreach(crrNode => {
        dictionaryTrie.merge(crrNode)
      })
    })


    this
  }


}

object LMDictionary {

  def apply(): LMDictionary = {
    val params = new Params()
    val dictionary = new LMDictionary()
    if (dictionary.exists()) dictionary.load()
    else dictionary
      .fromDictionary(params)
      .fromText(params).save()
  }

  def main(args: Array[String]): Unit = {


    val dictionary = apply()
    val items = dictionary.inference("yaklaşan")
    dictionary.filter(items, 3).foreach(println)
  }
}