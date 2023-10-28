package sampling.methods.core

import sampling.data.{Instance, TextInstance}

import java.util.Locale
import scala.io.Source

class WordExtractor(dictionarySize:Int, val wordRegex:String="[abcçdefgğhıijklmnoöprsştuüvyzqwxABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ]+", val dictionaryTextFilename:String = "resources/dictionary/lexicon.txt") extends Extractor() {

  var staticSet = readSet()
  def readSet(): Array[String] = {
    println("Constructing dictionary from text lexicon...")
    val locale = new Locale("tr")
    val array = Source.fromFile(dictionaryTextFilename).getLines()
      .map(line => line.split("\t").head.toLowerCase(locale)
        .replaceAll("[\\~\\&\\_]", "")
        .replaceAll("([au])([bcçdfgğhjklmnprsştvyz])E$", "$1$2a")
        .replaceAll("([eü])([bcçdfgğhjklmnprsştvyz])E$", "$1$2e")
        .toLowerCase(locale).trim).toArray
      .sortBy(_.length)
      .reverse


    array
  }

  override def process(instance: Instance, startIndex: Int): TextInstance = {
    val crrInstance = instance.asInstanceOf[TextInstance]
    val tokens = instance.asInstanceOf[TextInstance].text.split("[\\s\\p{Punct}\\p{Sc}\\p{Sm}]")
      .filter(token=> token.matches(wordRegex))
      .flatMap(token => {
        staticSet.find(item=> token.startsWith(item)) match{
          case Some(item) => Array(token, token.substring(0, item.length), token.substring(item.length))
          case None => token.sliding(5, 5).toArray
        }}).filter(_.nonEmpty)


    val maxSize = math.min(dictionarySize - 1, startIndex + dictionary.size)
    tokens.foreach(token => dictionary = dictionary.updated(token.hashCode, dictionary.getOrElse(token.hashCode, maxSize)))
    val featureSeq = Array(tokens.map(token => token.hashCode).toSeq)
    val features = tokens.map(token => (dictionary(token.hashCode).toInt -> 1d)).toMap
    //println("Dictionary size: "+limitSize)
    crrInstance.addFeatureSeq(featureSeq)
    crrInstance.setFeatures(features)

    crrInstance
  }

  override def itemize(instance: Instance): TextInstance = {
    process(instance, 0)
  }
}
