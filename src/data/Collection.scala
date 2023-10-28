package data

import smile.math.MathEx

import scala.util.Random

class Collection(var name: String, var array: Array[Embedding], var indices: Array[Int] = Array()) {

  def newCollection(crrArray:Array[Embedding]):Collection={
    new Collection(name, crrArray)
  }

  def innerEuclid(): Double = {
    val scores = array.flatMap(crr => array.map(other => crr.euclidean(other)))
    scores.sum / scores.length
  }

  def intersection(collection: Collection):Collection={
    val data = array.filter(crr=> collection.contains(crr))
    new Collection(name, data)
  }

  def contains(embedding: Embedding): Boolean = {
    array.contains(embedding)
  }

  def normalize(n:Double):Array[Array[Double]]={
    array.map(embedding=> embedding.array.map(p=> p + n))
  }

  def toArray():Array[Array[Double]]={
    array.map(_.array)
  }

  def sample(dim:Int):Array[Array[Double]]={
    array.map(_.array.slice(0, dim))
  }

  def transpose():Array[Array[Double]]={
    val matrix = Array.fill[Array[Double]](dim())(Array[Double]())
    for(i<-0 until array.length){
      for(j<-0 until dim()){
        matrix(j) = matrix(j) :+ array(i).array(j)
      }
    }
    matrix
  }

  def head():Embedding={
    array.head
  }


  def size():Int={
    array.length
  }

  def dim():Int = {
    array.head.embeddingSize
  }

  def innerCosine(): Double = {
    val scores = array.flatMap(crr => array.map(other => crr.cosine(other)))
    scores.sum / scores.length
  }

  def mean(): Embedding = {
    val score = MathEx.colMeans(array.map(item => item.array))
    new Embedding("mean", score)
  }

  def meanNormalize():this.type ={
    val meanVector = mean()
    array = array.map(embedding=> embedding.diff(meanVector))
    this
  }

  def select(index: Array[Int]): Collection = {
    val selectedArray = array.map(embedding => embedding.select(index))
    new Collection(name, selectedArray, index)
  }

  def cosine(collection: Collection): Array[Double] = {
    val scores = array.flatMap(crr => collection.array.map(other => crr.cosine(other)))
    scores
  }

  def cosineMean(collection: Collection): Double = {
    val scores = cosine(collection)
    scores.sum / scores.length
  }

  def covariance(): Array[Array[Double]] = {
    val matrix = array.map(item => item.array)
    MathEx.cov(matrix)
  }

  def random(other: Collection, sampleSize: Int, componentSize: Int): Array[Int] = {
    val mean1 = mean()
    val mean2 = other.mean()
    val samples = random(mean1, mean2, sampleSize, componentSize).head
    samples.indices
  }

  def random(mean: Embedding, other: Embedding, sampleSize: Int, componentSize: Int = 3, collectionSize: Int = 1000): Array[Collection] = {
    val randomlySelected = randomSubSelect(componentSize, array.head.embeddingSize, collectionSize)
      .sortBy(collection => other.select(collection.indices).cosine(collection) / mean.select(collection.indices).cosine(collection)).take(sampleSize)
    randomlySelected
  }

  def randomSelect(componentSize: Int, embeddingSize: Int): Array[Int] = {
    Range(0, componentSize).map(_ => Random.nextInt(embeddingSize)).toArray
  }

  def randomSubSelect(componentSize: Int, embeddingSize: Int, collectionSize: Int): Array[Collection] = {
    Range(0, collectionSize).map(_ => {
      randomSelect(componentSize, embeddingSize)
    }).map(indices => {
        val subArray = array.map(embedding => new Embedding(embedding.word, indices.map(i => embedding.array(i))))
        new Collection("subArray = " + indices.mkString("[", ",", "]"), subArray, indices)
      }).toArray
  }

  def diff(collection: Collection): Collection = {
    //difference between two collections
    val pairs = array.zip(collection.array).map(pair=> pair._1.diff(pair._2))
    new Collection(name, pairs)
  }
  def diffSimple(collection: Collection): Collection = {
    //difference between two collections
    val pairs = array.zip(collection.array).map(pair=> pair._1.diffSimple(pair._2))
    new Collection(name, pairs)
  }

  def klDivergence(collection: Collection): Double = {
    0.0
  }

  def union(collection:Collection): Collection = {
    val newArray = array ++ collection.array
    new Collection("[" + name + " union "+ collection.name + "]", newArray)

  }

  def show(): Unit = {
    array.foreach(embedding => {
      println(embedding.toString)
    })
  }

  def showItems(): Unit = {
    println("Collection: "+name)
    array.foreach(embedding=> println(embedding.word))
  }

  def showSimilarity(): Unit = {
    array.foreach(crr => {
      array.foreach(other => {
        println(crr.word + " ~ " + other.word + ": [" + crr.cosine(other) + "]")
      })
    })
  }
}

object Collection {

  def empty(test:String):Collection={
    new Collection(test, Array())
  }

  def negativeCollection(dictionary: Map[String, Embedding]): Collection = {
    val array = Array("gelmiyor", "gitmiyor", "silmiyor", "yapmıyor", "düşünmüyor", "yaşamıyor", "onaylamıyor", "söylemiyor","atamadı","atamıyor","atmadı",
      "anlamıyor", "bulunmuyor", "satılmıyor", "bilinmiyor", "almıyor", "görmüyor", "yardım etmiyor","gelmedi","bilemedi", "yaşamadı",
      "görme","yaşama","anlama","bilme","gezme","dalma","çalışma","bakma","yaşama","açma","kapama","zarflama","yazma","okuma","koşma","atlama",
      "vurma","zamlama","arttırma","azaltma")
    val crr = array.filter(word => dictionary.contains(word)).map(word => {
      dictionary(word)
    })
    new Collection("negative", crr)
  }

  def positiveCollection(dictionary: Map[String, Embedding]): Collection = {
    val array = Array("geliyor", "gidiyor", "siliyor", "yapıyor", "düşünüyor", "yaşıyor", "onaylıyor", "söylüyor","atadı","atıyor","attı",
      "anlıyor","bulunuyor","satılıyor","biliniyor","alıyor","görüyor","yardım ediyor","geldi","bildi","yaşadı", "gör", "görün","duy","duydu","onayladı",
      "onayla", "yap","bil","yaşa","gez","çalış", "zarfla","koş","atla","arttır","azalt","zamla","gazla","hızlan","yavaşla","vur","aç","kapa","yaz","oku","bak","dal","vardı","ulaştı","katıl","katıldı",
      "katılıyor", "gör","yaşa","tartıştı","duydu","bildi", "tartış","değiştir","duyur","anla","hisset","dokun","tat","yürü","zamanla","aldır","al","öğret","eğit","eğittir","öğren","çalış",
      "ek","dik","çömel","çömeliyor","koşuyor","görüyor","işle","işledi","tamamlıyor","haberleş","yarış", "oluşturdu",  "haberleşiyor",
      "durdur","gitti","durdurdu","tamamladı", "kalk","ata","yaklaş","beğen", "beğendi")
    val crr = array.filter(word => dictionary.contains(word)).map(word => {
      dictionary(word)
    })
    new Collection("positive", crr)
  }

  def verbCollection(dictionary: Map[String, Embedding]): Collection = {
    val array = Array("gel", "git", "sil", "yap", "düşün", "yaşa", "onayla", "söyle","ata","at",
      "anla","bulun","satıl","bilin","alıyor","görüyor","yardım et","geldi","bil","yaşa", "gör", "görün","duy",
      "onayla", "yap","bil","yaşa","gez","çalış", "zarfla","koş","atla","arttır","azalt","zamla","gazla","hızlan","yavaşla","vur","aç","kapa","yaz","oku","bak","dal","var","ulaş","katıl","katıl",
      "katıl", "gör","yaşa","tartış","duy","bil", "tartış","değiştir","duyur","anla","hisset","dokun","tat","yürü","zamanla","aldır","al","öğret","eğit","eğit","öğren","çalış",
      "ek","dik","çömel","koş","gör","işle","tamamla","yarış", "oluştur",  "haberleş",
      "git","durdur","tamamla", "kalk","ata","yaklaş","beğen")
    val crr = array.filter(word => dictionary.contains(word)).map(word => {
      dictionary(word)
    })
    new Collection("positive", crr)
  }



  def positiveNegativeCollection(dictionary: Map[String, Embedding]): (Collection,Collection) = {
    val source = Array("geliyor", "gidiyor", "siliyor", "yapıyor", "düşünüyor", "yaşıyor", "onaylıyor", "söylüyor","atadı","atıyor","attı",
      "anlıyor","bulunuyor","satılıyor","biliniyor","alıyor","görüyor","yardım ediyor","geldi","bildi","yaşadı")
    val destination = Array("gelmiyor", "gitmiyor", "silmiyor", "yapmıyor", "düşünmüyor", "yaşamıyor", "onaylamıyor", "söylemiyor", "atamadı", "atamıyor", "atmadı",
      "anlamıyor", "bulunmuyor", "satılmıyor", "bilinmiyor", "almıyor", "görmüyor", "yardım etmiyor", "gelmedi", "bilemedi", "yaşamadı")

    val crr = source.zip(destination).filter(pair => dictionary.contains(pair._1) && dictionary.contains(pair._2)).map(pair => {
      (dictionary(pair._1), dictionary(pair._2))
    })
    (new Collection("positive", crr.map(_._1)), new Collection("negative", crr.map(_._2)))

  }

  def nonVerbCollection(dictionary: Map[String, Embedding]):Collection={
    val array = Array("uzun", "yavaş", "zaman", "ev", "arava",
      "gelişim","erişim", "hızlı","bilgi", "otobüs","doğal", "şehir","akış", "yatak","algoritma", "dere","zamane", "çay")
    val crr = array.filter(word => dictionary.contains(word)).map(word => {
      dictionary(word)
    })
    new Collection("test", crr)
  }
  def testCollection(dictionary: Map[String, Embedding]): Collection = {
    val array = Array("uçtu","düşünüyor","yüzüyor","açıklanamıyor","zorlanmıyor",
      "tırmanıyor", "atıyor","atamıyor", "turlamadı", "yetişemedi", "çıkamadı", "gözükmüyor")
    val crr = array.filter(word => dictionary.contains(word)).map(word => {
      dictionary(word)
    })
    new Collection("test", crr)
  }

  def testSentiment(dictionary:Map[String, Embedding]):Array[EmbeddingRel] = {
    val array = Array("biliyor","bilmiyor","uçtu","uçmadı","yüzüyor","yüzmüyor","zorlanıyor","zorlanmıyor", "atıyor","atmıyor", "turladı","turlamadı", "güzüküyor", "gözükmüyor",
      "sevindi", "sevinmedi","bildi","bilmedi","anladı","anlamadı","anlıyor","anlamıyor","gördü","görmedi","görüyor","görmüyor","siliyor","silmiyor","yazıyor","yazmıyor", "sarıyor","sarmıyor",
      "alıyor","almıyor","seviyor","sevmiyor","yaşıyor","yaşamıyor","dalıyor","dalmıyor","daldı","dalmadı","kararıyor","kararmıyor","zarflıyor","zarflamıyor",
      "düşündü","düşünmedi","benimsedi","benimsemedi","yedi","yemedi","yakıyor","yakmıyor","soğutuyor","soğutmuyor","ısıtıyor","ısıtmıyor","ısınıyor","ısınmıyor",
      "hızlandı","hızlanmadı","yavaşlıyor","yavaşlamıyor","yavaşladı","yavaşlamadı","karışıyor","karışmıyor","çaldı","çalmadı","koşuyor","koşmuyor",
      "çalıyor", "çalmıyor","gitti","gitmedi","sürdü","sürmedi")
    array.sliding(2, 2).filter(array=> array.forall(dictionary.contains(_)))
      .map{case(Array(pos, neg))=> new EmbeddingRel("sentiment", dictionary(pos), dictionary(neg))}
      .toArray
  }

  def sampleIndices(collection: Collection, minSubject: Collection, precision: Double, size: Int): Collection = {
    val indices = collection.random(minSubject, 1, 4)
    collection.select(indices)
  }

  def main(args: Array[String]): Unit = {
    val map = Dataset.loadHuewei()

    val negativeSet = negativeCollection(map.embeddingMap)
    val positiveSet = positiveCollection(map.embeddingMap)

    println("Selection inner positives: " + negativeSet.innerCosine())
    println("Selection inner negatives: " + positiveSet.innerCosine())
    println("Selection cosine to subject: " + negativeSet.cosineMean(positiveSet))

    val negatives = sampleIndices(negativeSet, positiveSet, 0.005, 1)
    val positives = positiveSet.select(negatives.indices)

    println("Selection inner positives: " + positives.innerCosine())
    println("Selection inner negatives: " + negatives.innerCosine())
    println("Selection cosine to subject: " + positives.cosineMean(negatives))
  }
}
