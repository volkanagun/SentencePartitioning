package sampling.methods.committee

import sampling.data.TextInstance

class KLScorer(dictionarySize:Int, embeddingSize:Int, window: Int, committee: Int) extends VEScorer(dictionarySize, embeddingSize, window, committee) {

  var consensus = Map[Int, Node](0 -> Node(-1))

  override def score(instance: TextInstance): Double = {
    val textInstance = instance
    var total = 0d;

    textInstance.featureSequence.foreach(crr => {
      crr.sliding(window, 1).foreach(words =>{
        var previous = -1
        var sum = 0d
        models.foreach(model => {
          words.zipWithIndex.foreach { case (crr, indice) => {
            val mainScore = score(consensus, indice, previous, crr)
            val crrScore = score(model, indice, previous, crr)
            val klscore = mainScore * math.log((mainScore + 1) / crrScore)
            val finalScore = klscore / models.length
            sum = sum + finalScore
            previous = crr
          }
          }
        })

        total = total + 1.0 / textInstance.featureSequence.length * sum
      })

    })

    total
  }


  override def add(textInstance: TextInstance): KLScorer.this.type = {
    super.add(textInstance)
    consensus = add(consensus, textInstance)
    this
  }

  override def init(instances: Array[TextInstance]): KLScorer.this.type = {
    instances.foreach(instance=> add(instance))
    this
  }


}
