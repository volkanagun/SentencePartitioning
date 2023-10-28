package data

object Relations {

  var dictionary = Dataset.loadHuewei().embeddingMap

  def construct(name:String, array:Array[String]):Array[EmbeddingRel]={
    array.sliding(2,2).filter {case(Array(src, dst))=>{
      dictionary.contains(src) && dictionary.contains(dst)
    }}.map{case(Array(src, dst)) => {
      new EmbeddingRel(name, dictionary(src), dictionary(dst))
    }}.toArray
  }

  def collection(name: String, array:Array[EmbeddingRel]):Collection={
    val newArray = array.flatMap(rel=> Array(rel.src, rel.dst))
    new Collection(name, newArray)
  }

  def is_a():Array[EmbeddingRel] = {
    val arr = Array("kaplan","memeli", "yunus","memeli","köpek","memeli","timsah","sürüngen","tornavida","alet","ağaç","bitki",
      "çiçek", "bitki","yeşil","renk","saat","aksesuar","kolye","takı","pulbiber","baharat",
      "avukat","meslek")

    construct("is_a", arr)
  }

  def has_a():Array[EmbeddingRel] = {
    val arr = Array[String]("aslan","yele","insan","baş","hayvan","gövde","saat","yelkovan","araba","teker")
    construct("has_a", arr)
  }



}
