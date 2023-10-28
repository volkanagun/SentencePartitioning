package sampling.methods.committee

import sampling.data.TextInstance

class VotedDivergence(dictionarySize: Int, embeddingSize: Int, window: Int, committee: Int) extends VEScorer(dictionarySize, embeddingSize, window, committee) {
  //language model
  //Divergence between the committe distributions

  override def add(textInstance: TextInstance): this.type = {


    val allScores = models.zipWithIndex.flatMap { case (srcModel, i) => models.zipWithIndex.map { case (dstModel, j) => (srcModel, dstModel, i, j) } }
      .filter(tuple => tuple._3 > tuple._4)
      .map(tuple => (tuple._1, tuple._2, tuple._3))
      .map {
        case (source, target, i) => {
          var total = 0d;
          textInstance.featureSequence.foreach(seq => {
            var previous = -1
            seq.sliding(window, 1).foreach(words => {
              var sum = 0d

              words.zipWithIndex.foreach { case (crr, indice) => {
                val crrScore = score(source, target, indice, previous, crr)
                sum = sum + crrScore
                previous = crr
              }}

              total = total + -1.0 / textInstance.featureSequence.length * sum
            })
          })
          (total, i)
        }
      }

    val (crrScore, i) = allScores.sortBy(_._1).last
    models(i) = add(models(i), textInstance)
    this
  }

  override def score(textInstance: TextInstance): Double = {

    var total = 0d;

    textInstance.featureSequence.foreach(seq => {
      var previous = -1
      seq.sliding(window, 1).foreach(words => {
        var sum = 0d
        models.zipWithIndex.flatMap { case (srcModel, i) => models.zipWithIndex.map { case (dstModel, j) => (srcModel, dstModel, i, j) } }
          .filter(tuple => tuple._3 > tuple._4)
          .map(tuple => (tuple._1, tuple._2))
          .foreach {
            case (source, target) => {
              words.zipWithIndex.foreach { case (crr, indice) => {
                val crrScore = score(source, target, indice, previous, crr)
                sum = sum + crrScore
                previous = crr
              }
              }
            }
          }

        total = total + 1.0 / textInstance.featureSequence.length * sum
      })
    })

    total
  }

  def score(source: Map[Int, Node], target: Map[Int, Node], index: Int, previous: Int, crr: Int): Double = {

    if (source.contains(index) && target.contains(index) &&
      source(index).word == previous && target(index).word == previous) {
      val pxScore = source(index).score(crr)
      val qxScore = target(index).score(crr)
      pxScore * math.log(pxScore / qxScore)
    }
    else {
      1d
    }

  }
}
