package models;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Adam;
import org.bytedeco.pytorch.AdamOptions;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.InputArchive;
import org.bytedeco.pytorch.LSTMImpl;
import org.bytedeco.pytorch.LSTMOptions;
import org.bytedeco.pytorch.LinearImpl;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.OutputArchive;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.bytedeco.pytorch.LongOptional;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.File;
import java.util.Set;

import static org.bytedeco.pytorch.global.torch.argmax;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.cross_entropy_loss;
import static org.bytedeco.pytorch.global.torch.kCUDA;
import static org.bytedeco.pytorch.global.torch.kFloat;
import static org.bytedeco.pytorch.global.torch.kLong;
import static org.bytedeco.pytorch.global.torch.mean;
import static org.bytedeco.pytorch.global.torch.relu;

public final class StorchExtrinsicTrainer {

    private static final Set<String> CUDA_EXTRINSIC_TASKS = Set.of("ner", "pos", "sentiment");

    private StorchExtrinsicTrainer() {
    }

    public static Metrics trainAndEvaluate(
            String classifier,
            MultiDataSetIterator trainIterator,
            MultiDataSetIterator testIterator,
            int epochs,
            double learningRate,
            int embeddingLength,
            int hiddenLength,
            int labelCount,
            String modelFilename,
            int storchBatch
    ) {
        if (!CUDA_EXTRINSIC_TASKS.contains(classifier)) {
            throw new IllegalArgumentException(
                    "Torch CUDA extrinsic trainer only supports NER, POS, and Sentiment: " + classifier);
        }
        Device device = cudaDevice();
        System.err.println("Torch CUDA evaluation | task: " + classifier + " | device: cuda:0");
        StorchModule model = classifier.equals("sentiment")
                ? new SelfAttentionClassifier(embeddingLength, hiddenLength, labelCount)
                : new BiLstmSequenceClassifier(embeddingLength, hiddenLength, labelCount);

        File modelFile = new File(modelFilename + ".storch.pt");
        model.to(device, kFloat(), false);

        if (modelFile.exists()) {
            load(model, modelFile, device);
        } else {
            try {
                train(model, trainIterator, epochs, learningRate,
                        classifier.equals("sentiment"), device, storchBatch);
            } catch (Throwable error) {
                System.err.println();
                System.err.println("Torch training failed | task: " + classifier
                        + " | device: cuda:0"
                        + " | error: " + error.getClass().getName()
                        + ": " + String.valueOf(error.getMessage()));
                error.printStackTrace(System.err);
                System.err.flush();
                rethrow(error);
            }
            save(model, modelFile);
        }

        return evaluate(model, testIterator, classifier.equals("sentiment"), device);
    }

    private static void rethrow(Throwable error) {
        if (error instanceof RuntimeException) {
            throw (RuntimeException) error;
        }
        if (error instanceof Error) {
            throw (Error) error;
        }
        throw new IllegalStateException("Unexpected checked Torch training error", error);
    }

    private static void train(
            StorchModule model,
            MultiDataSetIterator iterator,
            int epochs,
            double learningRate,
            boolean sentenceClassifier,
            Device device,
            int storchBatch
    ) {
        Adam optimizer = new Adam(model.parameters(), new AdamOptions(learningRate));
        for (int epoch = 0; epoch < epochs; epoch++) {
            long batchCount = 0;
            long sampleCount = 0;
            long startedAt = System.currentTimeMillis();
            printTrainProgress(epoch, epochs, batchCount, sampleCount, storchBatch, startedAt);
            model.train(true);
            while (iterator.hasNext()) {
                org.nd4j.linalg.dataset.api.MultiDataSet batch = iterator.next();
                Tensor logits = model.forward(batchInput(batch, sentenceClassifier, device));
                Tensor targets = targets(batch.getLabels(0), sentenceClassifier, device);
                Tensor loss = loss(logits, targets, sentenceClassifier);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                batchCount++;
                sampleCount += batchSize(batch);
                printTrainProgress(epoch, epochs, batchCount, sampleCount, storchBatch, startedAt);
            }
            iterator.reset();
            System.err.println();
        }
    }

    private static Metrics evaluate(StorchModule model, MultiDataSetIterator iterator, boolean sentenceClassifier, Device device) {
        model.eval();
        long correct = 0;
        long total = 0;
        long[] truePositives = null;
        long[] falsePositives = null;
        long[] falseNegatives = null;
        while (iterator.hasNext()) {
            org.nd4j.linalg.dataset.api.MultiDataSet batch = iterator.next();
            Tensor logits = model.forward(batchInput(batch, sentenceClassifier, device));
            if (truePositives == null) {
                int labelCount = Math.toIntExact(logits.size(sentenceClassifier ? 1 : 2));
                truePositives = new long[labelCount];
                falsePositives = new long[labelCount];
                falseNegatives = new long[labelCount];
            }
            Tensor targets = targets(batch.getLabels(0), sentenceClassifier, device);
            Tensor predictions = sentenceClassifier
                    ? argmax(logits, new LongOptional(1), false)
                    : argmax(logits, new LongOptional(2), false).reshape(new long[]{-1});
            Tensor expected = sentenceClassifier ? targets : targets.reshape(new long[]{-1});

            Tensor predictionsCpu = predictions.cpu().contiguous();
            Tensor expectedCpu = expected.cpu().contiguous();
            long count = expectedCpu.numel();
            for (long i = 0; i < count; i++) {
                int predictedLabel = Math.toIntExact(predictionsCpu.data_ptr_long().get(i));
                int expectedLabel = Math.toIntExact(expectedCpu.data_ptr_long().get(i));
                if (predictedLabel == expectedLabel) {
                    correct++;
                    truePositives[expectedLabel]++;
                } else {
                    falsePositives[predictedLabel]++;
                    falseNegatives[expectedLabel]++;
                }
            }
            total += count;
        }
        iterator.reset();

        double accuracy = total == 0 ? 0d : (double) correct / (double) total;
        Metrics metrics = classificationMetrics(
                accuracy,
                truePositives,
                falsePositives,
                falseNegatives,
                sentenceClassifier ? 0 : 1);
        System.err.printf(
                "Torch evaluation | precision: %.6f | recall: %.6f | macro-F1: %.6f | accuracy: %.6f%n",
                metrics.precision,
                metrics.recall,
                metrics.f1,
                metrics.accuracy);
        return metrics;
    }

    private static Metrics classificationMetrics(
            double accuracy,
            long[] truePositives,
            long[] falsePositives,
            long[] falseNegatives,
            int firstLabel
    ) {
        if (truePositives == null) {
            return new Metrics(accuracy, 0d, 0d, 0d);
        }

        double precisionSum = 0d;
        double recallSum = 0d;
        double f1Sum = 0d;
        int evaluatedLabels = 0;
        for (int label = firstLabel; label < truePositives.length; label++) {
            long truePositive = truePositives[label];
            long falsePositive = falsePositives[label];
            long falseNegative = falseNegatives[label];
            long support = truePositive + falsePositive + falseNegative;
            if (support > 0L) {
                long precisionDenominator = truePositive + falsePositive;
                long recallDenominator = truePositive + falseNegative;
                long f1Denominator = 2L * truePositive + falsePositive + falseNegative;
                precisionSum += precisionDenominator == 0L
                        ? 0d
                        : (double) truePositive / precisionDenominator;
                recallSum += recallDenominator == 0L
                        ? 0d
                        : (double) truePositive / recallDenominator;
                f1Sum += f1Denominator == 0L
                        ? 0d
                        : (2d * truePositive) / f1Denominator;
                evaluatedLabels++;
            }
        }
        if (evaluatedLabels == 0) {
            return new Metrics(accuracy, 0d, 0d, 0d);
        }
        return new Metrics(
                accuracy,
                precisionSum / evaluatedLabels,
                recallSum / evaluatedLabels,
                f1Sum / evaluatedLabels);
    }

    private static Tensor loss(Tensor logits, Tensor targets, boolean sentenceClassifier) {
        if (sentenceClassifier) {
            return cross_entropy_loss(logits, targets);
        }
        return cross_entropy_loss(
                logits.reshape(new long[]{-1, logits.size(2)}),
                targets.reshape(new long[]{-1}));
    }

    private static StorchInput batchInput(org.nd4j.linalg.dataset.api.MultiDataSet batch, boolean sentenceClassifier, Device device) {
        if (sentenceClassifier) {
            return new StorchInput(toTorchRecurrent(batch.getFeatures(0), device), null);
        }
        return new StorchInput(toTorchRecurrent(batch.getFeatures(0), device), toTorchRecurrent(batch.getFeatures(1), device));
    }

    private static Tensor targets(INDArray labels, boolean sentenceClassifier, Device device) {
        Tensor labelTensor = toTorch(labels, device);
        return sentenceClassifier
                ? argmax(labelTensor, new LongOptional(1), false).to(device, kLong())
                : argmax(labelTensor, new LongOptional(1), false).to(device, kLong());
    }

    private static Tensor toTorch(INDArray array, Device device) {
        INDArray contiguous = array.castTo(DataType.FLOAT).dup('c');
        // The ND4J M2.1 ravel(char) implementation can route 'c' (ASCII 99)
        // into a reshape dimension. Flatten with an explicit long[] instead.
        float[] values = contiguous
                .reshape(new long[]{contiguous.length()})
                .toFloatVector();
        FloatPointer pointer = new FloatPointer(values);
        long[] shape = contiguous.shape();
        Tensor tensor = org.bytedeco.pytorch.global.torch
                .from_blob(pointer, shape, new TensorOptions(kFloat()))
                .clone();
        return tensor.to(device, kFloat());
    }

    private static Tensor toTorchRecurrent(INDArray array, Device device) {
        return toTorch(array.permute(0, 2, 1), device);
    }

    private static long batchSize(org.nd4j.linalg.dataset.api.MultiDataSet batch) {
        return batch.getLabels(0).shape()[0];
    }

    private static void printTrainProgress(
            int epoch,
            int epochs,
            long batchCount,
            long sampleCount,
            int storchBatch,
            long startedAt
    ) {
        long elapsedSeconds = (System.currentTimeMillis() - startedAt) / 1000L;
        System.err.print(String.format(
                "\rStorch epoch %d/%d | batches: %d | samples: %d | storchBatch: %d | elapsed: %ds",
                epoch + 1,
                epochs,
                batchCount,
                sampleCount,
                storchBatch,
                elapsedSeconds
        ));
        System.err.flush();
    }

    private static Device cudaDevice() {
        Device device = new Device(kCUDA(), (byte) 0);
        Tensor probe = org.bytedeco.pytorch.global.torch.rand(new long[]{1}, new TensorOptions(kFloat()));
        try {
            probe.to(device, kFloat());
        } catch (RuntimeException e) {
            throw new IllegalStateException("Storch extrinsic tasks require visible NVIDIA CUDA. Check nvidia-smi.", e);
        }
        return device;
    }

    private static void save(Module model, File file) {
        file.getParentFile().mkdirs();
        OutputArchive archive = new OutputArchive();
        model.save(archive);
        archive.save_to(file.getAbsolutePath());
    }

    private static void load(Module model, File file, Device device) {
        InputArchive archive = new InputArchive();
        archive.load_from(file.getAbsolutePath());
        model.load(archive);
        model.to(device, kFloat(), false);
    }

    public static final class Metrics {
        public final double accuracy;
        public final double precision;
        public final double recall;
        public final double f1;

        public Metrics(double accuracy, double precision, double recall, double f1) {
            this.accuracy = accuracy;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
        }
    }

    private static final class StorchInput {
        final Tensor leftOrInput;
        final Tensor right;

        StorchInput(Tensor leftOrInput, Tensor right) {
            this.leftOrInput = leftOrInput;
            this.right = right;
        }
    }

    private abstract static class StorchModule extends Module {
        StorchModule(String name) {
            super(name);
        }

        abstract Tensor forward(StorchInput input);
    }

    private static final class BiLstmSequenceClassifier extends StorchModule {
        private final LSTMImpl leftLstm;
        private final LSTMImpl rightLstm;
        private final LinearImpl classifier;

        BiLstmSequenceClassifier(int embeddingLength, int hiddenLength, int labelCount) {
            super("BiLstmSequenceClassifier");
            LSTMOptions leftOptions = new LSTMOptions(embeddingLength, hiddenLength);
            leftOptions.batch_first().put(true);
            LSTMOptions rightOptions = new LSTMOptions(embeddingLength, hiddenLength);
            rightOptions.batch_first().put(true);
            leftLstm = register_module("leftLstm", new LSTMImpl(leftOptions));
            rightLstm = register_module("rightLstm", new LSTMImpl(rightOptions));
            classifier = register_module("classifier", new LinearImpl(hiddenLength * 2L, labelCount));
        }

        @Override
        Tensor forward(StorchInput input) {
            Tensor left = input.leftOrInput;
            Tensor right = input.right;
            Tensor leftOut = leftLstm.forward(left).get0();
            Tensor rightOut = rightLstm.forward(right).get0();
            TensorVector tensors = new TensorVector(leftOut, rightOut);
            Tensor merged = cat(tensors, 2);
            return classifier.forward(merged);
        }
    }

    private static final class SelfAttentionClassifier extends StorchModule {
        private final LSTMImpl lstm;
        private final LinearImpl query;
        private final LinearImpl key;
        private final LinearImpl value;
        private final LinearImpl dense;
        private final LinearImpl classifier;

        SelfAttentionClassifier(int embeddingLength, int hiddenLength, int labelCount) {
            super("SelfAttentionClassifier");
            LSTMOptions options = new LSTMOptions(embeddingLength, hiddenLength);
            options.batch_first().put(true);
            lstm = register_module("lstm", new LSTMImpl(options));
            query = register_module("query", new LinearImpl(hiddenLength, hiddenLength));
            key = register_module("key", new LinearImpl(hiddenLength, hiddenLength));
            value = register_module("value", new LinearImpl(hiddenLength, hiddenLength));
            dense = register_module("dense", new LinearImpl(hiddenLength, hiddenLength));
            classifier = register_module("classifier", new LinearImpl(hiddenLength, labelCount));
        }

        @Override
        Tensor forward(StorchInput input) {
            Tensor sequence = input.leftOrInput;
            Tensor lstmOut = lstm.forward(sequence).get0();
            Tensor pooled = mean(lstmOut, new long[]{1}, false, null);
            Tensor hidden = relu(dense.forward(pooled));
            return classifier.forward(hidden);
        }
    }
}
