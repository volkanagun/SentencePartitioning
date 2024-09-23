package transducer

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.collection.immutable.HashSet
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.util.Random

case class Label(input: String, output: String, var count: Double) extends Serializable {

  var weight = 1d

  def load(in: ObjectInputStream): Label = {
    var iinput = in.readObject().asInstanceOf[String]
    var ooutput = in.readObject().asInstanceOf[String]
    var ccount = in.readDouble()
    Label(iinput, ooutput, ccount)
  }

  def save(out: ObjectOutputStream): this.type = {
    out.writeObject(input)
    out.writeObject(output)
    out.writeDouble(count)
    this
  }

  def inc(): Label = {
    count = count + 1
    this
  }

  def inc(value: Double): Label = {
    count = count + value
    this
  }

  def normalize(total: Double): Label = {
    weight = count / total
    this
  }


}

case class Target(id: Long, transition: Label) extends Serializable {

  def copy(): Target = {
    Target(id, Label(transition.input, transition.output, transition.count))
  }

  def load(in: ObjectInputStream): Target = {
    var jj = in.readLong()
    var ttransition = Label(null, null, 0).load(in)
    Target(jj, ttransition)
  }

  def save(out: ObjectOutputStream): Target = {
    out.writeLong(id)
    transition.save(out)
    this
  }

  def inc(): Target = {
    transition.inc()
    this
  }

  def incBy(value: Double): Target = {
    transition.inc(value)
    this
  }

  def count(): Double = {
    transition.count
  }

  def countLog(): Double = {
    Math.log(transition.count)
  }

  def logLikelihood(): Double = {
    Math.log(1.0 + transition.weight)
  }

  def normalize(total: Double): Target = {
    transition.normalize(total)
    this
  }

}

case class Score(var sequence: String, var score: Double) {

  def append(str: String, weight: Double): this.type = {
    sequence = sequence + str
    score += weight
    this
  }

  def entropy():Double={
    val split = sequence.split("[\\$\\#]")
    score / split.length
  }

  def copy(): Score = {
    Score(sequence, score)
  }

  override def hashCode(): Int = sequence.hashCode

  override def equals(obj: Any): Boolean = sequence.equals(obj.asInstanceOf[Score].sequence)
}

case class State(var i: Long, var stateMap: Map[String, Target] = Map()) extends Serializable {


  def copy(): State = {
   State(i, stateMap.map(pair => (pair._1 -> pair._2.copy())))
  }

  def mergeBy(ostate: State): State = {
    for (okey <- ostate.stateMap.keys) {
      val oval = ostate.stateMap(okey)
      if (stateMap.contains(okey)) {
        stateMap(okey).incBy(oval.transition.count)
      }
      else{
        stateMap = stateMap + (okey -> ostate.stateMap(okey))
      }
    }
    this
  }

  def load(in: ObjectInputStream): State = {
    val ii = in.readLong();
    val isize = in.readInt()
    var sstateMap = Map[String, Target]()
    for (i <- 0 until isize) {
      val key = in.readObject().asInstanceOf[String]
      val target = Target(0L, null).load(in)
      sstateMap = sstateMap.updated(key, target)
    }
    State(ii, sstateMap)
  }

  def save(out: ObjectOutputStream): State = {
    out.writeLong(i)
    out.writeInt(stateMap.size)
    stateMap.foreach { case (key, target) => {
      out.writeObject(key)
      target.save(out)
    }
    }

    this
  }

  def add(key: String, j: Long, transition: Label): Target = {
    synchronized {
      val target = stateMap.getOrElse(key, Target(j, transition))
      stateMap = stateMap.updated(key, target.inc())
      target
    }
  }

  def add(key: String, target: Target): Target = {
    synchronized {
      stateMap = stateMap.updated(key, target)
      target
    }
  }

  def has(key: String): Boolean = {
    stateMap.contains(key)
  }

  def normalize(states: Map[Long, State]): State = {
    val total = stateMap.toArray.par.map(_._2.count()).sum

    stateMap.toArray.par.foreach {
      case (key, transition) => {
        //val nstate = transition.normalize(total)
        transition.normalize(total)
        //if (states.contains(nstate.j)) states(nstate.j).normalize(states)
      }
    }

    this
  }

  def remove(set: Set[Long]): State = {
    stateMap.filter { case (key, target) => !set.contains(target.id) }
    this
  }

  def prune(deleteSet: Set[Long] = Set(), n: Int = 5): Set[Long] = {

    if (!stateMap.isEmpty && !deleteSet.contains(i)) {

      val sorted = stateMap.toArray.sortBy(_._2.count())
        .reverse

      val preserved = sorted.take(n)
      val removed = sorted.slice(n, sorted.length).map(_._2.id)
        .toSet

      stateMap = preserved.filter(k => !deleteSet.contains(k._2.id)).toMap

      removed

    }
    else {
      Set()
    }
  }

  override def hashCode(): Int = {
    var result = 7
    result = result * 31 + i.toInt
    result
  }

  override def equals(obj: Any): Boolean = {
    if (obj.isInstanceOf[State] && obj.asInstanceOf[State].i == i) true
    else false
  }
}

/**
 * @Author Dr. Hayri Volkan Agun
 * @Date 16.03.2022 19:58
 * @Project BigLanguage
 * @Version 1.0
 */

class Transducer extends Serializable {

  //a matrix and output
  var epsilon = "EPS"
  var marker = "#"
  var split = "[\\#\\$](\\s?)"
  var sep = "%%"
  var boundary = "$"
  var skip = "@"
  var skipBack = "-@"

  var map = Map[Long, State](0L -> State(0L))
  var finished = HashSet[Long]()


  def isEmpty():Boolean={
    map.size == 1 || map.isEmpty
  }

  def partition(d: Int): String = {
    if (d > 5) "dist" else if (d > 2) "neig" else "loc"
  }

  def copy(): Transducer = {
    val cpTrans = new Transducer()
    cpTrans.map = map.map(pair => (pair._1 -> pair._2.copy()))
    cpTrans.finished = finished
    cpTrans
  }

  def save(out: ObjectOutputStream): Transducer = {
    println("Saving transducer...")
    out.writeLong(map.size.toLong)
    map.foreach { case (id, state) => out.writeLong(id); state.save(out) }
    out.writeObject(finished)
    this
  }

  def stats():Map[String, Double]={
    var statMap = Map[String, Double]()
    val totalStates = map.size
    var totalTransitions = map.map(_._2.stateMap.size).sum
    val totalTransitionFreq = map.flatMap(_._2.stateMap.map(_._2.count())).sum

    statMap = statMap.updated("TotalStates", totalStates)
    statMap = statMap.updated("TotalTransitions", totalTransitions)
    statMap = statMap.updated("TotalTransitionFreq", totalTransitionFreq)

    statMap = statMap.updated("TransitionPerState", totalTransitions.toDouble/totalStates)
    statMap = statMap.updated("LogTransitionFreqPerTransition", math.log(totalTransitionFreq/totalTransitions))
    statMap = statMap.updated("LogTransitionFreqPerState", math.log(totalTransitionFreq/totalStates))

    statMap
  }

  def load(in: ObjectInputStream): Transducer = {
    println("Loading transducer...")
    val size1 = in.readLong()
    var i = 0L
    while (i < size1) {
      val id = in.readLong()
      if (!map.contains(id)) map = map + (id -> State(id).load(in))
      else {
        val newState = State(id).load(in)
        map(id).mergeBy(newState)
      }
      i += 1
    }

    finished = finished ++ in.readObject().asInstanceOf[HashSet[Long]]
    this
  }

  def hashLong(output: String): Long = {
    var result = 3L;
    for (i <- 0 until output.length) {
      result = output(i).toLong + result * 7
    }
    result;
  }

  def prune(n: Int = 5): Transducer = {
    var removed = Set[Long]()
    map.foreach { case (i, state) => removed = removed ++ state.prune(Set(), n) }
    map.foreach { case (i, state) => state.prune(removed, n) }
    map = map.filter { case (i, _) => !removed.contains(i) }
    this
  }


  def normalize(): Transducer = {
    map.toArray.par.foreach { case (_, state) => state.normalize(map) }
    this
  }


  def append(array: Array[Score], str: String, weight: Double): Array[Score] = {
    array.map(score => score.copy().append(str, weight))
  }

  def append(score: Score, str: String, weight: Double): Score = {
    score.copy().append(str, weight)
  }


  def addPrefix(i: Long, j: Long, input: Array[String]): State = {
    addPrefix(i, j, 1.0, input)
  }

  def addInput(i: Long, j: Long, input: String): State = {
    addPrefix(i, j, 1.0, Array(input, epsilon))
  }

  def addInput(input: Array[String], output: Array[String]): Unit = {
    var ii = 0L
    var j = 1L

    input.zip(output).foreach(pair => {
      val next = addPrefix(ii, j, Array(pair._1, pair._2))
      ii = next.i
      j = ii + 1
    })

    finished = finished + hashLong(output.mkString(""))
  }



  def addSkipEfficient(input: Array[Array[String]], distance: Int): Unit = {
    var ii = 0L
    var j = 1L

    input.zipWithIndex.par.foreach(item => {
      val crrIndex = item._2
      val crrCombination = item._1

      crrCombination.foreach(ngram => {
        val ngramSplits = ngram.split(split)
        ngramSplits.foreach(crrSplit=>{

          val next = addPrefix(ii, j, 1.0, Array(crrSplit, epsilon))
          var k = Math.max(0, crrIndex - distance)
          while (k > 0 &&  k < crrIndex) {
            val skipCombination = input(k)
            val skipDistance = partition(crrIndex - k)

            skipCombination.foreach(skipngram => {
              skipngram.split(split).foreach(skipSplit=>{
                val skipForward = skipSplit + skip + skipDistance
                addSkip(ii, next.i, 1.0, Array(skipForward, epsilon))
              })
            })

            k += 1
          }
          ii = next.i
          j = ii + 1
        })

      })
    })
  }

  def addSkip(input: Array[String], distance: Int): Unit = {
    var ii = 0L
    var j = 1L

    synchronized {
      input.zipWithIndex.foreach(pair => {
        val next = addPrefix(ii, j, 1.0, Array(pair._1, epsilon))
        for (k <- Math.max(0, pair._2 - distance) until pair._2) {
          val skipForward = input(k) + skip + partition(pair._2 - k)
          addSkip(ii, next.i, 1.0, Array(skipForward, epsilon))

        }
        ii = next.i
        j = ii + 1
      })

      finished = finished ++ input.map(item => hashLong(item))
    }
  }


  def addSkip(i: Long, j: Long, weight: Double, input: Array[String]): State = {
    val key = input(0)
    val transition = Label(input(0), input(1), weight)
    val statei = map.getOrElse(i, State(i))

    synchronized {
      map = map.updated(i, statei)
      val target = statei.add(key, j, transition)
      val statej = map.getOrElse(target.id, State(target.id))
      map = map.updated(target.id, statej)
      statej
    }
  }

  def addPrefix(i: Long, j: Long, weight: Double, input: Array[String]): State = {

    val key = input(0)
    val transition = Label(input(0), input(1), weight)
    val statei = map.getOrElse(i, State(i))

    map = map.updated(i, statei)
    val target = statei.add(key, map.size, transition)
    val statej = map.getOrElse(target.id, State(target.id))
    map = map.updated(target.id, statej)
    statej

  }

  def addPrefix(input: String): Unit = {
    var ii = 0L
    var j = 1L

    synchronized {
      input.foreach(character => {
        val next = addPrefix(ii, j, 1.0, Array(character.toString, epsilon))
        ii = next.i
        j = ii + 1
      })

      finished = finished + hashLong(input)
    }

  }

  def addPrefix(input: Array[String]): Unit = {
    var ii = 0L
    var j = 1L

    synchronized {
      input.foreach(item => {
        val next = addPrefix(ii, j, 1.0, Array(item, epsilon))
        ii = next.i
        j = ii + 1
      })

      finished = finished ++ input.map(item => hashLong(item))
    }
  }

  def addPrefix(input: Array[String], output: String): Unit = {
    var ii = 0L
    var j = 1L

    synchronized {
      input.foreach(item => {
        val next = addPrefix(ii, j, 1.0, Array(item, output))
        ii = next.i
        j = ii + 1
      })

      finished = finished ++ input.map(item => hashLong(item))
    }
  }

  def addPrefixes(inputs: Array[String]): Unit = {
    inputs.foreach(input => addPrefix(input))
  }

  def merge(other: Transducer): Transducer = {
    val otherMap: Map[Long, State] = other.map

    for (otherKey <- otherMap.keys) {
      val otherVal = otherMap(otherKey)
      if (map.contains(otherKey)) {
        val crrVal = map(otherKey)
        crrVal.mergeBy(otherVal)
      }
      else {
        map = map + (otherKey -> otherVal)
      }
    }

    for (id <- other.finished) {
      finished = finished + id
    }

    this
  }


  def end(str: String): Boolean = {
    finished.contains(hashLong(str))
  }


  def nextTransition(i: Long, input: String): Option[Target] = {

    if (map(i).has(input)) {
      val next = map(i).stateMap(input)
      Some(next)
    }
    else {
      None
    }
  }

  def hasTransition(i: Long, input: String): Boolean = {
    (map(i).has(input))
  }
/*

  def totalCount(i: Long): Double = {
    (map(i).stateMap.toArray.map(_._2.count()).sum)
  }

  def normCount(i: Long): Double = {
    (map(i).stateMap.toArray.map(_._2.count()).sum)
  }
*/

  def nextTransition(i: Long, input: String, label: String): Option[Target] = {

    if (map(i).has(input)) {
      val next = map(i).stateMap(input)
      if (next.transition.output.equals(label)) {
        Some(next)
      }
      else None
    }
    else {
      None
    }
  }

  def current(i: Long): State = {
    if (map.contains(i)) map(i)
    else {
      State(i)
    }
  }

  def next(i: Long, input: String): Option[State] = {
    if (map(i).has(input)) {
      val nextj = map(i).stateMap(input).id
      Some(map(nextj))
    }
    else {
      None
    }
  }

  def weight(i: Long, input: String): Double = {
    if (map(i).has(input)) {
      val nextLabel = map(i).stateMap(input)
      nextLabel.logLikelihood()
    }
    else {
      0d
    }
  }

  def next(i: Long, input: String, output: String): Option[State] = {
    val key = input + sep + output
    if (map(i).has(key)) {
      val nextj = map(i).stateMap(key).id
      Some(map(nextj))
    }
    else {
      None
    }
  }

  def skipSearch(input: Array[String], distance: Int): Double = {
    var current = map(0L)
    var i = 0;
    var total = 0d
    var scores = Range(0, input.length)

    scores.foreach(i => {

      var (j, sum) = loglikelihood(current.i, input(i))
      val min = Math.max(0, i - distance)

      for (k <- min until i) {
        val d = i - k
        val key = input(k) + skip + d
        val (_, sumSkip) = loglikelihood(current.i, key, j)
        sum = sum + sumSkip
      }

      current = map.getOrElse(j, map(0))
      total += sum

    })

    val norm = input.length * input.length
    total / norm
  }

  def rankingForward(crrMap:Map[Long, Double], input:Array[String], windowDistance:Int, iter:Int):Map[Long, Double]={
    var current = map(0L)
    var i = 0;
    var scoreMap = crrMap
    var scores = Range(0, input.length)

    var count = 0;
    while (count < iter) {

      scores.foreach(i => {

        val (j, sum) = loglikelihood(current.i, input(i))
        var jsum = 0.85f * scoreMap(current.i) + 0.15f * sum;
        val min = Math.max(0, i - windowDistance)
        for (k <- min until i) {
          val d = i - k
          val key = input(k) + skip + d
          val (_, sumSkip) = loglikelihood(current.i, key, j)
          jsum += 0.85f * scoreMap(current.i) + 0.15f * sumSkip;
        }

        jsum = jsum / iter
        scoreMap = scoreMap.updated(j, jsum)
        current = map.getOrElse(j, map(0))
      })

      count += 1
    }

    scoreMap
  }


  def rankingBackward(crrMap:Map[Long, Double], input:Array[String], windowDistance:Int, iter:Int):Map[Long, Double]={
    var current = map(0L)
    var i = 0;
    var scoreMap = crrMap
    var scores = Range(0, input.length)

    var count = 0;
    while (count < iter) {

      scores.foreach(i => {

        val (j, sum) = loglikelihood(current.i, input(i))
        var jsum = 0.85f * scoreMap(current.i) + 0.15f * sum;
        val min = Math.max(0, i - windowDistance)
        for (k <- min until i) {
          val d = i - k
          val key = input(k) + skipBack + d
          val (_, sumSkip) = loglikelihood(j, key, current.i)
          jsum += 0.85f * scoreMap(current.i) + 0.15f * sumSkip;
        }

        jsum = jsum / iter
        scoreMap = scoreMap.updated(j, jsum)
        current = map.getOrElse(j, map(0))
      })

      count += 1
    }

    scoreMap
  }


  def rankingSearch(input: Array[String], distance: Int, iter: Int): Double = {
    var scoreMap = Map[Long, Double](0L -> 1d)
    scoreMap = rankingForward(scoreMap, input, distance, iter)
    scoreMap = rankingBackward(scoreMap, input, distance, iter)
    val splitNorm = input.length * input.length
    scoreMap.map(_._2).sum / splitNorm
  }


  def loglikelihood(index: Long, current: String, ii: Long): (Long, Double) = {

    if (hasTransition(index, current)) {
      val next = nextTransition(index, current).get
      if (next.id == ii) (next.id, next.logLikelihood())
      else (0L, 0d)
    }
    else {
      (0L, 0d)
    }
  }

  def loglikelihood(index: Long, current: String): (Long, Double) = {

    if (hasTransition(index, current)) {
      val next = nextTransition(index, current).get
      (next.id, next.logLikelihood())
    }
    else {
      (0L, 0d)
    }
  }

  def likelyhoodSearch(input: Array[String]): Double = {
    var current = map(0L)
    var i = 0;
    var sum = 0d

    while (i < input.length) {

      current = (if (hasTransition(current.i, input(i))) current else map(0L))
      if (hasTransition(current.i, input(i))) {
        val next = nextTransition(current.i, input(i)).get
        if (map.contains(next.id)) {
          current = map(next.id)
        }
        else {
          current = map(0L)
        }
        sum += next.logLikelihood()
      }


      i += 1
    }

    sum

  }

  def produceOutput(input: Array[String]): Array[String] = {
    var current = map(0L)
    var i = 0;
    var result = Array[String]()

    while (i < input.length) {

      current = (if (hasTransition(current.i, input(i))) current else map(0L))
      if (hasTransition(current.i, input(i))) {
        val next = nextTransition(current.i, input(i)).get
        if (map.contains(next.id)) {
          current = map(next.id)
        }
        else {
          current = map(0L)
        }
        result :+= next.transition.output
      }
      i += 1
    }

    result
  }

  def likelyOutputSearch(input: Array[String], output: Array[String], iin: Long = 0L, currentin: State = map(0L), arrin: Array[String] = Array(""), twice: Boolean = false): Double = {

    var arr = arrin;
    var current = currentin
    var result = Array[String]()
    var i = 0;
    var sum = 0d

    while (i < input.length) {

      val ch = input(i) + sep + output(i)
      val out = output(i)
      val next = nextTransition(current.i, ch, out)
      if (next.isDefined) {
        current = map(next.get.id)
        sum -= next.get.countLog()
        i = i + 1
      }
      else if (i == 0) return 0;
      else {
        current = map(0)
      }
    }

    sum
  }


  def tokenEntropySplit(input: String, top: Int, sample:Int = 1): Array[String] = {
    val search = tokenSplitEntropy(input).take(top)
    val result = Random.shuffle(search.toSeq).take(sample).toArray
    result
  }

  def tokenSplit(input: String, top: Int, sample:Int): Array[String] = {
    val search = tokenSearch(input.toCharArray.map(_.toString))
      .sortBy(_.entropy())
      .reverse.take(top)
      .map(_.sequence)

    val random = new Random(17).shuffle(search.toSeq)
    val result = random.take(sample).toArray
    result
  }

  def tokenSplit(input: String, top: Int): Array[String] = {
    val search = tokenSearch(input.toCharArray.map(_.toString))
    val result = search.sortBy(_.score)(Ordering[Double].reverse).take(top)
      .map(_.sequence)
    result
  }

  def multipleSplitSearch(input: Array[String], top: Int): Array[String] = {
    val search = tokenSearch(input)
    val result = search.sortBy(_.score)(Ordering[Double].reverse).take(top)
      .map(_.sequence)
    result
  }

  def suffixSplitSearch(input: String, top: Int): Array[String] = {
    val search = tokenSearch(input.toCharArray.map(_.toString))
    val result = search.sortBy(_.score)(Ordering[Double].reverse).take(top)
      .map(_.sequence)
    val suffixation = result.map(token => {
      val subitems = token.split(split)
      subitems.head + marker + subitems.tail.mkString("")
    })

    suffixation
  }

  def tokenSplitEntropy(input: String): Array[String] = {
    val found = tokenSearch(input.toCharArray.map(_.toString)).map(item => {
        val sequence = item.sequence
        val score = item.score
        val sp = sequence.split(split).filter(item => item.nonEmpty)
        val ep = sp.filter(item => end(item)).length + 1d
        (sequence, score *  ep / sp.length)
      }).sortBy(_._2).reverse
      .map(item => item._1)

    found
  }

  def tokenSearch(input: Array[String], iin: Int = 0, currentin: State = map(0), arrin: Array[Score] = Array(Score("", 0d))): Array[Score] = {
    val jjn = iin + 1
    val nexti = next(currentin.i, input(iin))
    var result = Array[Score]()

    if (nexti.isDefined && jjn < input.length) {
      val weighti = weight(currentin.i, input(iin))
      val crr = input.slice(0, jjn).mkString
      val nonmarked = append(arrin, input(iin), weighti)
      result = result ++ tokenSearch(input, jjn, nexti.get, nonmarked)
      if (end(crr)) {
        val marking = append(arrin, input(iin) + marker, weighti)
        result = result ++ tokenSearch(input, jjn, map(0), marking)
      }
    }
    else if (!nexti.isDefined && jjn < input.length) {
      val ninput = input.slice(jjn, input.length)
      val njjn = 0

      val marking = append(arrin, input(iin), 0d)
      result = result ++ tokenSearch(ninput, njjn, map(0), marking)
    }
    else {
      result = append(arrin, input(iin) + boundary, 1d)
    }

    result
  }

  def longestSearch(input: String): String = {

    var previous = false
    var current = map(0)
    var str = ""
    var chars = input.toCharArray
    var i = 0
    while (i < input.length) {
      val ch = chars(i).toString
      val state = next(current.i, ch)

      if (state.isDefined) {

        if (end(str + ch) && previous) {
          str = str + ch;
        }
        else if (end(str) && !previous) {
          str = str + marker + ch;
        }
        else if (end(str)) {
          str = str + marker + ch
        }
        else {
          str = str + ch
        }
        current = state.get
        i = i + 1
        previous = true
      }
      else if (str.endsWith(marker)) {
        current = map(0)
        str = str + ch
        i = i + 1
        previous = false
      }
      else if (previous) {
        current = map(0)
        str = str + marker + ch
        i = i + 1
        previous = false
      }
      else {
        str = str + ch
        i = i + 1
        current = map(0)
        previous = false
      }
    }

    str
  }

  def longestSearch(input: Array[String]): String = {

    var current = map(0)
    var str = ""
    var chars = input
    var i = 0
    while (i < input.length) {
      val ch = chars(i)
      val state = next(current.i, ch)

      if (state.isDefined) {
        if (end(str + ch)) {
          str = str + ch;
        }
        else if (end(str)) {
          str = str + marker + ch
        }
        else {
          str = str + ch
        }
        current = state.get
        i = i + 1
      }
      else if (str.endsWith(marker)) {
        current = map(0)
        str = str + ch
        i = i + 1
      }
      else {
        current = map(0)
        str = str + marker
      }
    }

    str
  }

}
