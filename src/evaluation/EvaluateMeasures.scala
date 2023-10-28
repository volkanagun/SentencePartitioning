package evaluation

class EvaluateMeasures {

  case class CountKey(item:String, score:Float){
    override def hashCode(): Int = item.hashCode()
    override def equals(obj: Any): Boolean = item.equals(obj.asInstanceOf[CountKey].item)
  }

  case class Count(original:String, predicted:String, var score:Float)
  {
    def increment():Count={
      score = score + 1
      this
    }
    def toActual():(String, Float)={
      (original, score)
    }

    def tpScore(actual:String):Float={
      if(original.equals(actual) && predicted.equals(original)) score
      else 0f
    }

    def fpScore(actual:String):Float={
      if(predicted.equals(actual) && !predicted.equals(original)) score
      else 0f
    }

    def fnScore(actual:String):Float={
      if(original.equals(actual) && !predicted.equals(original)) score
      else 0f
    }

    def tnScore(actual:String):Float={
      if(!original.equals(actual) && predicted.equals(original)) score
      else 0f
    }

  }

  var totalSampleCount:Long = 0
  var totalPatternCount:Long = 0
  var confusionMatrix = Map[String, Array[Count]]()

  def count():Map[String, Float] = {
    confusionMatrix.view.mapValues(_.map(_.score).sum)
      .toMap
  }

  def measures():Map[String, Array[CountKey]] = {
    confusionMatrix.view.map{case(actual, array)=> {
      val tp = array.map(_.tpScore(actual)).sum
      val fp = array.map(_.fpScore(actual)).sum
      val fn = array.map(_.fnScore(actual)).sum
      val tn = array.map(_.tnScore(actual)).sum
      (actual -> Array(CountKey("true-positive", tp), CountKey("false-positive", fp), CountKey("false-negative", fn), CountKey("true-negative", tn)))
    }}.toMap
  }


}
