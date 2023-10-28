package sampling.data

abstract class Instance extends Serializable {
  var features:Map[Int, Double] = Map()

  def setFeatures(map:Map[Int, Double]):this.type ={
    this.features = map
    this
  }
}

class TextInstance(var text:String) extends Instance{
  var featureSequence = Array[Seq[Int]]()

  def addFeatureSeq(featureSequence : Array[Seq[Int]]):this.type ={
    this.featureSequence = this.featureSequence ++ featureSequence
    this
  }

  def setFeatureSeq(featureSequence : Array[Seq[Int]]):this.type ={
    this.featureSequence = featureSequence
    this
  }

  def setText(text:String):this.type ={
    this.text = text
    this
  }

  override def hashCode(): Int = text.hashCode()
}
