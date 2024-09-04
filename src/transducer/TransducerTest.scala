package transducer

object TransducerTest {

  def testDictionary(): Unit = {
    val transducer = new Transducer()
    val seq = Array("cam","camdan","camda", "camlar","camdaki")
    val top = 3
    transducer.addPrefixes(seq)
    transducer.multipleSplitSearch("camdakiler", top).foreach(item => println(item))
  }

  def main(args: Array[String]): Unit = {
    testDictionary()
  }
}
