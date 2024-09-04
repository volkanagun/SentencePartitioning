package tagging

import tagging.lemmatizer.RegexTokenizer

object SpanTest {
  def test():Unit = {

    var array = Array("Saat 11:52'de meydana gelen depremde",
      "100.000 kişi evsiz kaldı.")
    val tagger = RegexTokenizer()

    array.foreach(sentence => {
      println(sentence)
      println(tagger.process(sentence).map(span=> span.toString).mkString(","))
    })

  }

  def main(args: Array[String]): Unit = {
    test()
  }
}
