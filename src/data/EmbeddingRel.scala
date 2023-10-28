package data

class EmbeddingRel(val name:String, val src:Embedding, val dst:Embedding) {

  def srcDiff(pairEmbedding: EmbeddingRel):Embedding={
    pairEmbedding.src.diff(src)
  }
  def dstDiff(pairEmbedding: EmbeddingRel):Embedding={
    pairEmbedding.dst.diff(dst)
  }

  def dim():Int={
    src.embeddingSize
  }

  def diff():Embedding={
    dst.diff(src)
  }

  def cosine():Double={
    src.cosine(dst)
  }

  override def toString: String = "["+src.word + "~" + dst.word+"]"
}
