package sampling.methods.core

import sampling.data.{Instance, TextInstance}

class MultiExtractor(val array: Array[Extractor]) extends Extractor {
  override def process(instance: Instance, startIndex: Int): TextInstance = {

    var startIndex = 0;
    var map = Map[Int, Double]()
    var textInstance = instance.asInstanceOf[TextInstance]
    array.foreach(extractor => {
      val crrInstance = extractor.process(textInstance, startIndex)
      startIndex = extractor.dictionary.size
      crrInstance.features.foreach { case (index, score) => {
        map = map.updated(index, score)
      }}
    })
    textInstance
  }

  override def itemize(instance: Instance): TextInstance = process(instance, 0)
}

object MultiExtractor{

  def main(args: Array[String]): Unit = {
    val textInstance1 = new TextInstance("ali veli kirk dokuz elli")
    val textInstance2 = new TextInstance("aliler veliler kirk dokuz elliler")
    val textInstance3 = new TextInstance("zamane kediler elliler")

    val ngramExtractor = new NgramExtractor(100,5)
    val wordExtractor = new WordExtractor(1000)
    val readabilityExtractor = new ReadabilityExtractor()
    val multiExtractor = new MultiExtractor(Array(wordExtractor, readabilityExtractor, ngramExtractor))

    val map1 = multiExtractor.process(textInstance1, 0)
    show(map1.features)

    val map2 = multiExtractor.process(textInstance2, 0)
    show(map2.features)

    val map3 = multiExtractor.process(textInstance3, 0)
    show(map3.features)
  }

  def show(map:Map[Int, Double]): Unit = {
    val result = map.map{case(ii, score) => {
      ii + "->" + score
    }}.mkString("[",",","]")

    println(result)
  }
}
