package sampling.methods.core

import sampling.data.{Instance, TextInstance}

abstract class Extractor extends Serializable {

  var dictionary = Map[Int, Double]()
  var dictionarySize = 10000

  def setDictionarySize(dictionarySize: Int):this.type ={
    this.dictionarySize = dictionarySize
    this
  }

  def save() = this
  def load() = this

  def exists():Boolean = true


  def build(iterator:Iterator[TextInstance]) = this
  def parbuild(iterator:Iterator[TextInstance]) = this

  def process(instance:Instance, startIndex:Int) :TextInstance

  def itemize(instance:Instance):TextInstance

  def filter(instance:TextInstance):Boolean = {
    !instance.featureSequence.isEmpty && instance.featureSequence.forall(_.length >= 2)
  }

  def prune():this.type = {
    dictionary =  dictionary.toArray
      .sortBy(_._2).reverse
      .take(dictionarySize)
      .toMap

    this
  }
}
