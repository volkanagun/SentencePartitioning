package experiments

object LMParallel {

  def main(args: Array[String]): Unit = {

    new LMDataset().constructParallel()
  }
}
