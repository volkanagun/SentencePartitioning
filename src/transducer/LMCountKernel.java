package transducer;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.exception.AparapiException;
import com.aparapi.internal.kernel.KernelManager;

import java.util.Random;

public class LMCountKernel extends Kernel {
    private final int[][] sequences;
    private final int[][][] counts;
    private final int maxWindow;


    public LMCountKernel(int[][] sequences, int[][][] counts, int maxWindow) {
        this.sequences = sequences;
        this.counts = counts;
        this.maxWindow = maxWindow;
    }

    public int[][][] getCounts() {
        return counts;
    }

    public void count(int sid, int i, int j) {
        int isym = sequences[sid][i];
        int jsym = sequences[sid][j];
        int d = i - j;
        counts[isym][jsym][d] = counts[isym][jsym][d] + 1;
    }

    @Override
    public void run() {
        int sid = getGlobalId(0);
        int iid = getGlobalId(1);

        int j = iid - maxWindow + 1;
        while (j >= 0 && j < iid) {
            count(sid, iid, j);
            j = j + 1;
        }

    }

    public static Device[] list() {
        return KernelManager.instance().getDefaultPreferences()
                .getPreferredDevices(null).toArray(new Device[0]);
    }

    public static int[][] create(int windowSize) {
        int sampleSize = 100;
        int[][] sequence = new int[sampleSize][windowSize];
        for (int i = 0; i < sampleSize; i++) {
            for (int j = 0; j < windowSize; j++) {
                sequence[i][j] = 2;
            }
        }
        return sequence;
    }

    public static int[][][] execute(int[][] sequences, int maxDictionary, int maxWindow) {
        final int[][][] counts = new int[maxDictionary][maxDictionary][maxWindow];

        try {
            Device[] devices = list();
            Device device = devices[0];

            LMCountKernel kernel = new LMCountKernel(sequences, counts, maxWindow);
            kernel.run();//kernel.execute(Range.create2D(device, sequences.length,maxWindow, 1, 1));
            int[][][] result = kernel.getCounts();
            kernel.cleanUpArrays();
            kernel.dispose();
            return result;
        } catch (Throwable ex) {
            return counts;
        }
    }


}
