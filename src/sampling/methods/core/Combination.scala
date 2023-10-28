package sampling.methods.core

case class Combination(w1: Int, w2: Int) {
  override def hashCode(): Int = w1 * 7 + w2

  override def equals(obj: Any): Boolean = {
    val other = obj.asInstanceOf[Combination]
    w1 == other.w1 && w2 == other.w2
  }
}