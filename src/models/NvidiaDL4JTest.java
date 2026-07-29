package models;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Locale;
import java.util.stream.Collectors;

public final class NvidiaDL4JTest {

    private NvidiaDL4JTest() {
    }

    public static void main(String[] args) throws Exception {
        printNvidiaSmi();

        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        String backend = Nd4j.getBackend().getClass().getName();
        String executioner = Nd4j.getExecutioner().getClass().getName();

        System.out.println("ND4J backend: " + backend);
        System.out.println("ND4J executioner: " + executioner);

        requireCudaBackend(backend, executioner);
        runArraySmokeTest();
        runDl4jSmokeTest();

        System.out.println("NVIDIA DL4J test passed.");
    }

    private static void printNvidiaSmi() {
        try {
            Process process = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"
            ).redirectErrorStream(true).start();

            String output;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                output = reader.lines().collect(Collectors.joining(System.lineSeparator()));
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("nvidia-smi: " + output);
            } else {
                System.out.println("nvidia-smi failed: " + output);
            }
        } catch (IOException e) {
            System.out.println("nvidia-smi unavailable: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.out.println("nvidia-smi interrupted.");
        }
    }

    private static void requireCudaBackend(String backend, String executioner) {
        String combined = (backend + " " + executioner).toLowerCase(Locale.ROOT);
        if (!combined.contains("cuda") && !combined.contains("jcublas")) {
            throw new IllegalStateException(
                    "CUDA backend is not active. Run with Maven profile '-Pcuda' and make sure nvidia-smi works."
            );
        }
    }

    private static void runArraySmokeTest() {
        INDArray left = Nd4j.rand(DataType.FLOAT, 1024, 1024);
        INDArray right = Nd4j.rand(DataType.FLOAT, 1024, 1024);
        INDArray product = left.mmul(right);
        System.out.println("ND4J mmul checksum: " + product.sumNumber());
    }

    private static void runDl4jSmokeTest() {
        MultiLayerNetwork network = new MultiLayerNetwork(
                new NeuralNetConfiguration.Builder()
                        .seed(42)
                        .dataType(DataType.FLOAT)
                        .updater(new Adam(0.01))
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new DenseLayer.Builder()
                                .nIn(4)
                                .nOut(8)
                                .activation(Activation.RELU)
                                .build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(8)
                                .nOut(2)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .build()
        );

        network.init();

        INDArray features = Nd4j.createFromArray(new float[][]{
                {0f, 0f, 0f, 1f},
                {0f, 1f, 0f, 1f},
                {1f, 0f, 1f, 0f},
                {1f, 1f, 1f, 0f}
        });
        INDArray labels = Nd4j.createFromArray(new float[][]{
                {1f, 0f},
                {1f, 0f},
                {0f, 1f},
                {0f, 1f}
        });

        network.fit(new DataSet(features, labels));
        System.out.println("DL4J score: " + network.score());
    }
}
