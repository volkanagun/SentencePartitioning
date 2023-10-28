package data

import smile.math.MathEx


class Embedding(val word:String, var array:Array[Double]) {
  override def toString: String = word + ":" + array.mkString("[",",","]")

  def embeddingSize = array.length

  def contains(embedding:Embedding):Boolean={
    word.contains(embedding.word) && !word.equals(embedding.word)
  }

  def copy():Embedding = {
    new Embedding(word, array)
  }

  def transpose():Array[Array[Double]]={
    var newArray = Array[Array[Double]]()
    for(i<-0 until array.length){
      newArray = newArray :+ Array(array(i))
    }
    newArray
  }

  def indexOut(index:Int):Array[Double]={
    array.zipWithIndex.filter(_._2 != index)
      .map(_._1)
  }

  def indexValue(index:Int):Double={
    array(index)
  }
  def select(index:Array[Int]):Embedding={
    val narray = array.zipWithIndex.map{case(value, indice)=> {
      if(index.contains(indice)) value else 0.0
    }}
    new Embedding(word, narray)
  }

  def norm():Double={
    math.sqrt(array.map(f=> f*f).sum)
  }
  def normSquare():Double={
    array.map(f=> f*f).sum
  }

  def orthogonal(projection:Embedding): Embedding = {
    val qembedding = projection
    val dot = qembedding.dot(this)
    val scaled = qembedding.scale(dot)
    this.diff(scaled)
  }

  def klDivergence(other: Embedding): Unit = {
    MathEx.KullbackLeiblerDivergence(array, other.array)
  }
  def jesenDivergence(other: Embedding): Unit = {
    MathEx.JensenShannonDivergence(array, other.array)
  }

  def normalize():this.type = {
    val unitNorm = norm()
    array = array.map(id=> id/unitNorm)
    this
  }

  def scale(value:Double):Embedding={
    val narray = array.map(f => f*value)
    new Embedding(word, narray)
  }
  def diff(other:Embedding):Embedding={
    val newArray = array.zip(other.array).map(pair=> pair._1 - pair._2)
    val str = "["+word + "-"+other.word + "]"
    new Embedding(str, newArray)
  }

  def diffSimple(other:Embedding):Embedding={
    val newArray = array.zip(other.array).map(pair=> pair._1 - pair._2)
    new Embedding(word, newArray)
  }

  def dot(other:Embedding):Double={
    val newArray = array.zip(other.array).map(pair=> pair._1 * pair._2)
    newArray.sum
  }
  def mul(other:Embedding):Array[Array[Double]]={
    val matrix = Array.fill[Array[Double]](other.embeddingSize)(Array.fill[Double](other.embeddingSize)(0d))
    for(i<-0 until embeddingSize){
      for(j<-0 until other.embeddingSize){
        matrix(i)(j) = array(i) * other.array(j)
      }
    }
    matrix
  }

  def add(other:Embedding):Embedding={
    val newArray = array.zip(other.array).map(pair=> pair._1 + pair._2)
    val str = "["+word + "+>"+other.word + "]"
    new Embedding(str, newArray)
  }
  def add(other:Array[Double]):Embedding={
    val newArray = array.zip(other).map(pair=> pair._1 + pair._2)
    new Embedding(word, newArray)
  }
  def subtract(other:Embedding):Embedding={
    val newArray = array.zip(other.array).map(pair=> pair._1 - pair._2)
    val str = "["+word + "->"+other.word + "]"
    new Embedding(str, newArray)
  }

  def average(other:Embedding):Embedding={
    val newArray = array.zip(other.array).map(pair=> (pair._1 + pair._2)/2)
    val str = "["+word + "+>"+other.word + "]"
    new Embedding(str, newArray)
  }

  def cosineComplement(embedding: Embedding):Double={
    1 - cosine(embedding)
  }

  def cosine(embedding:Embedding):Double = {
    val dotProduct = embedding.array.zip(array).map(pair=> pair._1 * pair._2).sum
    dotProduct / (embedding.norm() * norm())
  }
  def cosine(embedding:Collection):Double = {
    embedding.array.map(other=> other.cosine(this)).sum / embedding.array.length
  }

  def euclidean(embedding: Embedding):Double={
    val nom = embedding.array.zip(array).map(pair=> math.pow(pair._1-pair._2, 2)).sum
    math.sqrt(nom)
  }


  override def hashCode(): Int = word.hashCode

  override def equals(obj: Any): Boolean = {
    obj.asInstanceOf[Embedding].word.equals(word)
  }
}



object Embedding{
  def zeros(name:String, size:Int):Embedding={
    val array = Array.fill[Double](size)(0d)
    new Embedding(name, array)
  }
  def zeros(name:String, size1:Int, size2:Int):Array[Embedding]={
    Array.tabulate[Embedding](size1)(i => Embedding.zeros(s"${name}_${i}", size2))
  }

  def avg(array:Array[Embedding]):Embedding={
    val embedding = zeros("avg",array.head.embeddingSize)
    val sum = array.foldRight[Embedding](embedding){case(item, main)=> main.add(item)}
    sum.scale(array.length)
  }

  def slide(array:Array[Array[Double]], window:Int):Array[Array[Embedding]] = {
    array.map(subarray => {
      subarray.sliding(window, 1).map(item => new Embedding("slide",item))
        .toArray
    })
  }
  def slideArray(array:Array[Double], window:Int):Array[Embedding] = {

      array.sliding(window, 1).map(item => new Embedding("slide",item))
        .toArray

  }
}