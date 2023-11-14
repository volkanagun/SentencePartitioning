package experiments
import scala.sys.process._
object SamplingCommand {
  def main(args: Array[String]): Unit = {
    val command = "java -jar -Xmx64G ActiveSelection.jar"
    for(i<-0 until 100) {
      val result: String = command.!!
    }
  }
}
