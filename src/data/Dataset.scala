package data

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.io.Source

class Dataset(val filename:String) {
  var embeddingMap = Map[String, Embedding]()
  def add(word:String, array:Array[Double]):this.type ={
    embeddingMap = embeddingMap + (word-> new Embedding(word, array))
    this
  }
}

object Dataset{

  var huewei = "resources/embeddings/huawei-skipgram-min_count_10-word_dim_300.word2vec_format"
  var hueweiBinary = "resources/binary/huawei.bin"

  def loadHuewei():Dataset={
    load(hueweiBinary)
  }



  def save(dataset:Dataset, fname:String): Dataset = {
    val obj = new ObjectOutputStream(new FileOutputStream(fname))
    obj.writeInt(dataset.embeddingMap.size)
    dataset.embeddingMap.foreach{case(item, embedding)=>{
      obj.writeObject(item)
      obj.writeObject(embedding.array)
    }}
    obj.close()
    dataset
  }

  def load(fname:String): Dataset = {
    val dataset = new Dataset(fname)
    val obj = new ObjectInputStream(new FileInputStream(fname))
    val size = obj.readInt()

    for(i<-0 until size){
      val item = obj.readObject().asInstanceOf[String]
      val array = obj.readObject().asInstanceOf[Array[Double]]
      dataset.add(item, array)
    }

    obj.close()
    dataset
  }

  def loadText(filename:String, splitter:String = "[\\s\\t]"):Dataset={
    println("Loading embedding from text")
    val dataset = new Dataset(filename)
    Source.fromFile(filename).getLines().drop(1).foreach(line=>{
      val splitLine = line.split(splitter)
      val words = splitLine.takeWhile(item=> item.matches("[\\p{P}\\p{L}ıüğçş`,,\\p{S}]+"))
      if(words.nonEmpty) {
        val array = splitLine.drop(words.length).map(value => value.toDouble)
        dataset.add(words.mkString(" "), array)
      }
    })
    println("Loading finished...")
    dataset
  }

  def main(args: Array[String]): Unit = {
    loadHuewei()
  }

}
