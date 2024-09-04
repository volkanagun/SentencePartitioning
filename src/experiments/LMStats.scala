package experiments

class LMStats {

  var totalSentenceCount = 0
  var totalTokenCount = 0
  var totalTime = 0L

  var startTime = 0L
  var endTime = 0L

  var statMap = Map[String, Double]()

  def setGraphStatistics(map:Map[String, Double]):this.type ={
    statMap = map
    this
  }

  def avgTokenTime():Double = {
    totalTime.toDouble / totalTokenCount
  }
  def avgSentenceTime():Double = {
    totalTime.toDouble / totalSentenceCount
  }

  def incSentenceCount(count:Int):this.type ={
    totalSentenceCount+=count
    this
  }
  def incTokenCount(count:Int):this.type ={
    totalTokenCount+=count
    this
  }

  def start():this.type ={
    startTime = System.currentTimeMillis()
    this
  }

  def end():this.type ={
    endTime = System.currentTimeMillis()
    totalTime += endTime-startTime
    this
  }
}
