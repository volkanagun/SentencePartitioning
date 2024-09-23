package helper

class LMScore(var count:Long, var score:Double) extends Serializable{

  def incrementBy(value:Long = 1L): this.type = {
    count+= value
    this
  }

  def normalize(total:Long):this.type ={
    score = count.toDouble/total
    this
  }
}
