package sampling.methods.statistical

import sampling.data.TextInstance
import sampling.methods.core.InstanceScorer

class VoabSelect(dictionarySize: Int, embeddingSize: Int) extends InstanceScorer(dictionarySize, embeddingSize) {
  override def score(instance: TextInstance): Double = random.between(0.0, 10000.0)
  override def add(instance: TextInstance): VoabSelect.this.type = this
  override def init(instances: Array[TextInstance]): VoabSelect.this.type = this

}
