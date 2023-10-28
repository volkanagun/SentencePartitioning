package sampling.methods.core

import sampling.data.{Instance, TextInstance}

class ReadabilityExtractor() extends Extractor {

  var nonvovels = "[bcçdfgğhjklmnprsştvyz]"
  var vovels ="[aeıioöuü]"
  var syllables=s"(${nonvovels}${vovels}${nonvovels}${nonvovels}|" +
    s"${vovels}${nonvovels}${nonvovels}|" +
    s"${nonvovels}${vovels}${nonvovels}|" +
    s"${nonvovels}${vovels}|" +
    s"${vovels})"


  override def itemize(instance: Instance): TextInstance = process(instance, 0)

  override def process(instance: Instance, startIndex: Int): TextInstance = {
    val crrInstance = instance.asInstanceOf[TextInstance]
    val tokens = crrInstance.text.split("\\s+")
    val someratios = ratios(tokens)
    val map = someratios.zipWithIndex.map(pair => (startIndex + pair._2 -> pair._1))
      .toMap

    val seq = map.map(pair=> math.floor(pair._2).toInt).toSeq

    map.foreach(pair=> dictionary = dictionary.updated(pair._1, pair._2))
    crrInstance.setFeatures(map)
    crrInstance.addFeatureSeq(Array(seq))
  }

  def ratios(tokens:Array[String]):Array[Double] = {
    val tokenSeq = tokens.sortBy(token=> token.length)
    val tokenSize = tokenSeq.length
    val totalTokenLength = tokens.map(token=> token.length).sum.toDouble
    val avgTokenLength = totalTokenLength / tokens.length
    val totalPunctuations = tokens.filter(token=> token.matches("\\p{Punct}+")).size.toDouble
    val avgPunctuations = totalPunctuations / tokens.length
    val repeats = tokenSeq.map(token => tokenSeq.filter(_.contains(token)).length).sum.toDouble
    val avgrepeats = repeats / tokens.length
    val syllableCount = tokens.map(token=> syllables(token)).sum.toDouble
    val avgsyllableCount = syllableCount/tokens.length

    Array[Double](tokenSize, totalTokenLength, totalTokenLength, avgTokenLength, totalPunctuations,
      avgPunctuations,
      repeats,
      avgrepeats,
      syllableCount,
      avgsyllableCount)
  }

  def syllables(token:String): Int = {
    syllables.r.findAllIn(token).length
  }

  override def filter(instance: TextInstance): Boolean = {
    instance.featureSequence.nonEmpty && instance.featureSequence.head.nonEmpty
  }
}
