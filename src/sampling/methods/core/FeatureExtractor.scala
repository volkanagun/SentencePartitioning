package sampling.methods.core

import sampling.data.{Instance, TextInstance}

import java.io._
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable

class FeatureExtractor(dictionarySize: Int, secondDictionarySize: Int, val ngramSize: Int, val freq: Int = 20) extends Extractor() {

  var mapSecond = Map[Int, Int]()

  override def exists(): Boolean = {
    new File("resources/binary/features.bin").exists()
  }

  override def save(): Extractor = {
    println("Saving FeatureExtractor")
    val stream = new ObjectOutputStream(new FileOutputStream("resources/binary/features.bin"))
    stream.writeInt(dictionary.size)
    dictionary.foreach { case (i, d) => stream.writeInt(i); stream.writeDouble(d) }
    stream.writeInt(mapSecond.size)
    mapSecond.foreach { case (ik, d) => stream.writeInt(ik); stream.writeInt(d) }
    stream.close()
    this
  }

  override def load(): Extractor = {
    println("Loading FeatureExtractor")
    val stream = new ObjectInputStream(new FileInputStream("resources/binary/features.bin"))

    val sz1 = stream.readInt()
    for (i <- 0 until sz1) {
      dictionary = dictionary + (stream.readInt() -> stream.readDouble())
    }

    val sz3 = stream.readInt()
    for (i <- 0 until sz3) {
      mapSecond = mapSecond + (stream.readInt() -> stream.readInt())
    }
    stream.close()
    this
  }

  def merge(other: FeatureExtractor): this.type = {
    other.dictionary.foreach { case (id1, cnt) => {
      dictionary = dictionary.updated(id1, dictionary.getOrElse(id1, 0d) + cnt)
    }}

    other.mapSecond.foreach { case (id1, id2) => {
      mapSecond = mapSecond.updated(id1, mapSecond.getOrElse(id1, id2))
    }}

    this
  }

  override def parbuild(iterator: Iterator[TextInstance]): this.type = {
    iterator.sliding(120000, 120000).foreach(collection => {
      collection.sliding(5000, 5000).toArray.par.map(sequence => {
        new FeatureExtractor(dictionarySize, secondDictionarySize, ngramSize, freq).build(sequence.iterator)
      }).toArray.foreach(fe => merge(fe))
    })

    this
  }

  override def build(iterator: Iterator[TextInstance]): this.type = {
    println("Building...")
    iterator.foreach(instance => itemize(instance))

    dictionary = dictionary.filter { case (_, count) => count > freq }
    dictionary = dictionary.toArray.sortBy(_._2).reverse.take(dictionarySize).toMap
    dictionary.keys.foreach{id=>{
      mapSecond = mapSecond.updated(id, mapSecond.size)
    }}

    this
  }

  override def itemize(instance: Instance): TextInstance = {
    val crrInstance = instance.asInstanceOf[TextInstance]
    val tokens = crrInstance.text.split("[\\p{Punct}\\p{S}\\d\\s]")
      .map(token=> token.substring(0, math.min(token.length, token.length)))
    tokens.foreach(token => {
      dictionary = dictionary.updated(token.hashCode, dictionary.getOrElse(token.hashCode, 0d) + 1d)
    })

    val featureSeq = Array(tokens.map(token => token.hashCode).toSeq)
    val features = tokens.map(token => {
      if(!dictionary.contains(token.hashCode)) (dictionarySize-1 -> 1d)
      else token.hashCode -> 1d
    }).toMap

    crrInstance.setFeatureSeq(featureSeq)
    crrInstance.setFeatures(features)

  }

  override def process(instance: Instance, startIndex: Int): TextInstance = {
    val processed = itemize(instance)
    processed.features = processed.features.map {
      case (ik, _) => (mapSecond.getOrElse(ik, secondDictionarySize - 1) -> 1.0)
    }
    processed
  }
}
