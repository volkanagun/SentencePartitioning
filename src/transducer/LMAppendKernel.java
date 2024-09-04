package transducer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;

public class LMAppendKernel extends Kernel {
    private int maxDictionary;
    private int[][][][] counts;
    private int[][][] result;


    public LMAppendKernel(int[][][][] counts, int maxDictionary, int maxWindow) {
        this.maxDictionary = maxDictionary;
        this.counts = counts;
        this.result = new int[maxDictionary][maxDictionary][maxWindow];
    }

    public int getMaxDictionary() {
        return maxDictionary;
    }

    public void setMaxDictionary(int maxDictionary) {
        this.maxDictionary = maxDictionary;
    }

    public int[][][][] getCounts() {
        return counts;
    }

    public void setCounts(int[][][][] counts) {
        this.counts = counts;
    }

    public int[][][] getResult() {
        return result;
    }

    public void setResult(int[][][] result) {
        this.result = result;
    }

    @Override
    public void run() {
        int symii = getGroupId(0);
        int symjj = getGroupId(1);
        int dd = getGroupId(2);


        for (int i = 0; i < counts.length; i++) {
            int count = counts[i][symii][symjj][dd];
            result[symii][symjj][dd] += count;
        }

    }

    public static Device[] list() {
        return KernelManager.instance().getDefaultPreferences()
                .getPreferredDevices(null).toArray(new Device[0]);
    }


    public static int[][][] execute(int[][][][] counts, int maxDictionary, int maxWindow) {

        Device device = list()[1];
        LMAppendKernel kernel = new LMAppendKernel(counts, maxDictionary, maxWindow);
        kernel.execute(Range.create3D(device, maxDictionary, maxDictionary, maxWindow, 1, 1, 1));
        return kernel.getResult();
    }

    public static int[][][] execute(int[][][][] counts, int[][][] crr, int maxDictionary, int maxWindow) {
        System.setProperty("com.aparapi.executionMode", "GPU");
        Device device = list()[1];
        LMAppendKernel kernel = new LMAppendKernel(counts, maxDictionary, maxWindow);
        kernel.setResult(crr);
        kernel.execute(Range.create3D(device, maxDictionary, maxDictionary, maxWindow, 1, 1, 1));
        return kernel.getResult();
    }
}
