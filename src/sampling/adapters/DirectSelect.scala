package sampling.adapters

import sampling.data.TextInstance

class DirectSelect(maxSelectSize:Int) extends ScoreAdapter(maxSelectSize) {
  override def filter(array: Array[TextInstance]): Array[TextInstance] = {
    val selected = array.take(maxSelectSize - count.toInt)
    count += selected.length
    selected
  }

  override def init(array: Array[TextInstance]): DirectSelect.this.type = {
    this
  }
}
