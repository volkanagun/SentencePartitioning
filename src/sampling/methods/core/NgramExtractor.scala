package sampling.methods.core

import sampling.data.{Instance, TextInstance}

class NgramExtractor(dictionarySize:Int, val ngramSize:Int) extends Extractor() {
  override def process(instance: Instance, startIndex:Int): TextInstance = {
    val crrInstance = instance.asInstanceOf[TextInstance]
    val tokens = crrInstance.text.toCharArray.sliding(ngramSize, 3)
      .map(array=> array.mkString("")).toArray
    val maxSize = math.min(dictionarySize-1, startIndex+dictionary.size)
    tokens.foreach(token=> {dictionary = dictionary.updated(token.hashCode, dictionary.getOrElse(token.hashCode, maxSize))})
    val featureSeq = Array(tokens.map(token=> token.hashCode).toSeq)
    val features = tokens.map(token => (dictionary(token.hashCode).toInt -> 1d)).toMap
    crrInstance.addFeatureSeq(featureSeq)
    crrInstance.setFeatures(features)

  }

  override def itemize(instance: Instance): TextInstance = process(instance, 0)
}
