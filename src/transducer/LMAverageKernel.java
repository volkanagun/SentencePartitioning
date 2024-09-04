package transducer;

import com.aparapi.Kernel;
import com.aparapi.Range;

public class LMAverageKernel extends Kernel {
    private int[][][] counts;
    private float[][][] average;
    private int[][] sum;
    private int maxDictionary, maxWindows;
    public LMAverageKernel(int[][][] counts, int maxDictionary, int maxWindow) {
        this.counts = counts;
        this.sum = new int[maxDictionary][maxWindow];
        this.maxDictionary = maxDictionary;
        this.maxWindows = maxWindow;
    }

    public float[][][] getAverage() {
        return average;
    }

    @Override
    public void run() {
       int symii =  getGlobalId(0);
       int dd =  getGlobalId(1);

       for(int symjj=0; symjj<maxDictionary; symjj++){
           sum[symii][dd] = sum[symii][dd]  + counts[symii][symjj][dd];
       }

       for(int symjj=0; symjj<maxDictionary; symjj++){
           average[symii][symjj][dd] = ((float) counts[symii][symjj][dd]) / sum[symii][dd];
       }

    }



    public static float[][][] execute(int[][][] counts, int maxDictionary, int maxWindow){
        System.setProperty("com.aparapi.executionMode", "GPU");
        LMAverageKernel kernel = new LMAverageKernel(counts, maxDictionary, maxWindow);
        kernel.execute(Range.create3D(maxDictionary,maxDictionary, maxWindow));
        return kernel.getAverage();
    }
}
