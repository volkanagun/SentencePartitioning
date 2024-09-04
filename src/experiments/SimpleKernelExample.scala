package experiments

import com.aparapi.Kernel
import com.aparapi.device.Device
import com.aparapi.device.OpenCLDevice

object SimpleKernelExample extends App {
  // Enable Aparapi logging
  System.setProperty("com.aparapi.enableExecutionModeReporting", "true")
  System.setProperty("com.aparapi.enableProfiling", "true")
  System.setProperty("com.aparapi.enableVerboseJNI", "true")
  System.setProperty("com.aparapi.dumpProfilesOnExit", "true")

  // Set the Aparapi execution mode to GPU
  System.setProperty("com.aparapi.executionMode", "GPU")

  // Define a simple kernel
  val kernel = new Kernel() {
    override def run() = {
      val i = getGlobalId()
      // Simple operation: set output[i] to i
      output(i) = i
    }

    var output: Array[Int] = _
  }

  // Initialize the output array with a valid buffer size
  val bufferSize = 1024 // Set a valid buffer size
  kernel.output = new Array[Int](bufferSize)

  try {
    // Execute the kernel
    kernel.execute(bufferSize)

    // Print the output
    println("Kernel execution completed. Output:")
    kernel.output.foreach(i => print(s"$i "))
    println()
  } catch {
    case e: Exception =>
      println("Error during kernel execution: " + e.getMessage)
      e.printStackTrace()
  } finally {
    // Dispose of the kernel
    kernel.dispose()
  }
}
