package transducer

class RankGpuKey(val i:Int, val j:Int, val d:Int){
  override def hashCode(): Int = {
    var result = i
    result = result * 7 + j
    result = result * 7 + d
    result
  }

  override def equals(obj: Any): Boolean = {
    val other = obj.asInstanceOf[RankGpuKey]
    other.i == i && other.j == j && other.d == d
  }
}
