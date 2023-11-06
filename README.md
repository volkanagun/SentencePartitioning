# ActiveSelection

The project is maven project. The intellij is used as an ide. It requires JDK11 and Scala 2.13.8 in order to compile. Other than these two environments, Intel Oneapi 2023.2 may be required and LD_LIBRARY must be specified to locate the correct intelone api bindings. 

In ActiveSelection library several active selection approaches are implemented to extract representative sentences from a large text corpus. The code contains two experimental stages. In the first stage the dataset is constructed. 
In this stage, an active selection method uses an Z-score based online moving scoring average to obtain a score and decide on the informativeness of the target sentence. The sentences having sginificant scores are selected from this set and saved to a text file line by line.
In the second stage, a deep learning model is used to extract word embeddings from the selected sentences. Later, these embeddings are used to evaluate the quality of the embeddings. The active selection methods are compared with the same set of embedding extraction method and intrinsic/extrinsic evaluation dataset.
The evaluations are saved to resources/results folder along with the parameters. The parameters are used for hasing the result file. So, for each set of parameters such as extraction model, embedding vector length, and selection method, there should be a unique result file.   

# ExperimentSPL

ExperimentSPL is the main entry point for the first stage of the program. In ExperimentalSPL the parameters relevant to the selected active selection methods are used to create a dataset consists of selected sentences. 
ExperimentSPL requires a large line by line text corpus and a large word embedding binary. Before running ExperimentSPL both files must be placed within paths resources/text/sentences/sentences-tr.txt and resources/binary/huewei.bin respectively. A general overview of this processing pipeline is schematized as follows.

![ExperimentSPL processing steps](https://github.com/volkanagun/ActiveSelection/blob/master/pipeline.jpg?raw=true)

An important part of this processing pipeline is the construction of the static vocabulary corpus. In this step, the dictionary of the target evaluation tests are used to filter the sentences. In this respect, each target word is used to sample the corpus sentences randomly so that each word will have equal number of sentences. Note that the sampled sentences must be also distinct. In this procesing pipeline, before the selection process the dictionaries of the target evaluation datasets must be constructed. In order to construct the static dictionaries DatasetGeneration class must be ran first. The following main block represents the steps of DatasetGeneration.

```scala
object DatasetConversion extends DatasetConversion() {
  def main(args: Array[String]): Unit = {
    convertNER("resources/evaluation/ner/train.json","train")
    convertNER("resources/evaluation/ner/test.json","test")
    convertPOS("resources/evaluation/pos/boun-train.conllu","train")
    convertPOS("resources/evaluation/pos/boun-test.conllu","test")
    convertSentiment("resources/evaluation/sentiment/train.csv", "train")
    convertSentiment("resources/evaluation/sentiment/test.csv", "test")

    createNERVocabulary()
    createSentimentVocabulary()
    createPOSVocabulary()
  }
}
```
## Methods 

ExperimentSPL contains all the steps for selection of the sentences. These steps include feature extraction steps, selection method, scoring technique. These steps can be itemized as follows:

1. Feature extraction
   - Tokens : Uses a standard tokenizer and use tokens as features.
   - N-Grams: Uses a frequent n-gram dictionary to extract frequent n-grams from sentences.
2. Selection methods
   - VocabSelect : VocabSelect uses same amount of sentences for each word in the evaluation dataset. It selects these words randomly without using any criteria. 
   - KMeans : K-Means is a model based selection method. It uses an average to score the candidate sentence. The parameters k is selected as 1 to 10. If the sentence is very similar to mean of any of the cluster, it is discarded.
   - Least : Least sqaures method is a model based selection methods. It uses the Least squares regression to select the candidate sentence.
   - Hopfield : Hopfield neural network model based selection method. It uses energy of the hopfields to score the candidate sentence. 
   - Boltzmann: Hopfield neural network model based selection method. It uses energy of the Boltzman machine. Has a different neural network structure and learning algorithm comapred to Hopfield neural network. 
   - VotedDivergence: It uses a voting schema based on ranomly constructed voters. The divergence measures are applied for voting the candidate sentence.
   - VE: Uses next/skip word prediction based voting. Similar to VotedDivergence but includes local next word frequencies.
   - LM: Use a language model perplexity based scoring. Different from VE, it only uses peoplexity in next word predictions.
   - Mahalonabis: A distance metric. It uses Mahalonabis distance for embedding space of the sentences. If the similarity is high, the sentence is discarded.
   - Euclidean: A distance metric. It uses Euclidean distance to for the selected sentences. If the similarity is high, the sentence is discarded.
   - KL : A distance metric for distributions. It uses KL-divergence between the candidate sentence distribution from all the set and the distribution of the selected sentence. If the difference is low, the sentence is discarded. 
   - Entropy: It uses information entropy to select the sentences. It is similar to LM but uses consequtive words.
3. Scoring methods:
   - Average soring: The average selection scores of the previously selected sentences is used as a threshold for deciding on the selection of new candidate sentences.
   - Majority voting: The selection scores of set of selection methods are used as a majority voting schema. Only when the majority of the decisions selects the candidate sentence, then the sentence is selected.
  
Along with these selection choices, several other parameters such as embedding size, window length of the language model, maximum dictionary size, cluster size (k-nn) are stored in [SampleParams](https://github.com/volkanagun/ActiveSelection/blob/master/src/sampling/experiments/SampleParams.scala) class. The following functions inside [ExperimentSPL](https://github.com/volkanagun/ActiveSelection/blob/master/src/sampling/experiments/ExperimentSPL.scala) must be modified to include other scoring methods and feature extractors. 

## Scoring function
Must include new instance of InstanceScorer as a scoring function.
 ```scala
      def createCriterias(sampleName: String): InstanceScorer
   ```
## Feature extractor
Must include a feature extractor for modelling features. For instance for English or German a new tokenizer, stemmer, or other features can be defined as an extractor.
 ```scala
def createExtractor(name: String): Extractor 
 ``` 
## Adapter
Adapters can be defined here. Adapters are used for decision when a scoring method is given.
```scala 
def createAdapter(adapterName: String, scorer: InstanceScorer, k: Int, kselectSize: Int, maxSelectSize: Int, threshold: Double): ScoreAdapter
```

# Evaluation

How relevant the selected sentences and the selection methodology is a question that shouldbe answered through the second stage. In this stage, a deep learning model is used to construct word embeddings from selected sentences. The general overview of this stage is given in the following figure.

![SamplingExperiment steps](https://github.com/volkanagun/ActiveSelection/blob/master/evaluation.jpg?raw=true)

Word embedding extraction approaches can be diverse. In this software, Skip-Gram, CBOW and Self-Attention LSTM models are used. The extraction methods uses the sampled datasets to extract the word embeddings. The quality of the extracted word embeddings are evaluated through evaluation methods. In order to evaluate the word embeddings and sentence selection methods there must be an evaluation dataset. There are two types of evaluation datasets. These are intrinsic and extrinsic. Intrinsic evaluation tests word embeddings in analogy pairs based on cosine similarity. Extrinsic evaluation datasets uses word embeddings in a machine learning model and predicts the accurateness of the trained model on a test dataset. Extrinsic evaluation requires a training phase,however intrinsic evaluation just uses the extracted word embeddings. There are also two types extrinsic evaluation. These are sequential models, and classification models which are defined based on the type of machine learning model. Evaluation produces a score for the word embedding extraction process. The results are saved into results folder.

## Extraction Models

Three extraction models namely SkipGram, CBOW and Self-Attention LSTM models are applied to extract word embeddings from sampled datasets. To use custom embedding extraction models the class namely **EmbeddingModel** must be derived. The functions train(), save(), load() are used to train the word embeddings, save and load them into/from binary files. During loading and saving the **update(ngram: String, vector: Array[Float]): Int** function must be called. This function returns the indice of the ngram and stores the embedding of the n-gram.

## Evaluation mdoels

To evaluate other datasets with the current definitions, either a json dataset or a line by line text dataset is required. JSON files are used for intrinsic evaluations. A JSON dataset for English and German can be constructed easily from the sentence-tr.json example placed inside resources/evaluation/analogy folder. For extrinsic evaluation the example datasets are placed inside resources/evaluation/ folder.
  

   
