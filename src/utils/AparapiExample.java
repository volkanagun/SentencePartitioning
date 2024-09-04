package utils;

import com.aparapi.Kernel;
import com.aparapi.Range;

public class AparapiExample {
    public static void main(String[] args) {
        final int size = 1024;
        final int[] inputArray = new int[size];
        final int[] outputArray = new int[size];

        // Initialize the input array
        for (int i = 0; i < size; i++) {
            inputArray[i] = i;
        }

        // Create an Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int gid = getGlobalId();
                outputArray[gid] = inputArray[gid] * inputArray[gid];
            }
        };

        // Execute the kernel with the specified range
        kernel.execute(Range.create(size));

        // Print the results
        for (int i = 0; i < size; i++) {
            System.out.println("inputArray[" + i + "] = " + inputArray[i] + ", outputArray[" + i + "] = " + outputArray[i]);
        }

        // Dispose of the kernel resources
        kernel.dispose();
    }
}
