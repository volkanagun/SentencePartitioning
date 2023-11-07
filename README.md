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
   - N-Grams: Uses a frequent n-gram dictionary to extract frequent n-grams from sentences. This is a custom tokenizer that must be defined for each language separately.
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

Three extraction models namely SkipGram, CBOW and Self-Attention LSTM models are applied to extract word embeddings from sampled datasets. To use custom embedding extraction models the class namely **EmbeddingModel** must be derived. The functions train(), save(), load() are used to train the word embeddings, save and load them into/from binary files. During loading and saving the **update(ngram: String, vector: Array[Float]): Int** function must be called. This function returns the indice of the ngram and stores the embedding of the n-gram. The following code block shows the EmbeddingModel class.

```scala
abstract class EmbeddingModel(val params: SampleParams) extends IntrinsicFunction {

  var avgTime = 0d
  var sampleCount = 0

  var dictionaryIndex = Map[String, Int]("dummy" -> 0)
  var dictionary = Map[String, Array[Float]]("dummy" -> Array.fill[Float](params.embeddingLength)(0f))
  var computationGraph: ComputationGraph = null

  lazy val tokenizer = new Tokenizer().loadBinary()

  def getTrainTime(): Double = avgTime

  def train(filename: String): EmbeddingModel

  def save(): EmbeddingModel

  def load(): EmbeddingModel


  def getDictionary(): Map[String, Array[Float]] = dictionary

  def getDictionaryIndex(): Map[Int, Array[Float]] = {
    dictionary.map { case (ngram, vector) => dictionaryIndex(ngram) -> vector }
  }

  def update(ngram: String, vector: Array[Float]): Int = {
    dictionary = dictionary.updated(ngram, vector)
    update(ngram)
  }

  def update(ngram: String): Int = {

    if (dictionaryIndex.size < params.dictionarySize) {
      dictionaryIndex = dictionaryIndex.updated(ngram, dictionaryIndex.getOrElse(ngram, dictionaryIndex.size))
    }
    retrieve(ngram)

  }

  def retrieve(ngram: String): Int = {
    dictionaryIndex.getOrElse(ngram, 0)
  }

  def tokenize(sentence: String): Array[String] = {
    tokenizer.ngramFilter(sentence)
  }

  def forward(token: String): Array[Float] = {
    val frequentNgrams = tokenizer.ngramStemFilter(token).filter(ngram => dictionary.contains(ngram))
    val ngramVectors = frequentNgrams.map(ngram => dictionary(ngram))
    average(ngramVectors)
  }

  def average(embeddings: Array[Array[Float]]): Array[Float] = {
    var foldResult = Array.fill[Float](params.embeddingLength)(0f)
    embeddings.foldRight[Array[Float]](foldResult) { case (a, main) => {
      main.zip(a).map(pair => pair._1 + pair._2)
    }
    }
  }
}
```

In order to use a different types of tokenizers, the **forward(token:String)** method must be modified. In this settings, a language dependent tokenizer extracts the frequent n-grams of a sentence. In order to extract n-grams in other languages, another tokenizer must be used here. Note that forward method uses averaging for token n-grams. It assumes that the embedding of a token is the average of its n-gram content. EmbeddingModel also stores the embeddings for each string either in token or n-gram form. Along with the embedding vectors, the indice of the n-grams is also stored. Through the update and retrieve methods, the indices of a n-gram sequence can be converted to BOW or One-Hot sequence.      

## Evaluation models

To evaluate other datasets with the current definitions, either a json dataset or a line by line text dataset is required. JSON files are used for intrinsic evaluations. A JSON dataset for English and German can be constructed easily from the sentence-tr.json example placed inside resources/evaluation/analogy folder. For extrinsic evaluation the example datasets are placed inside resources/evaluation/ folder. Extrinsic evaluation models use Self-Attention recurrent model and implements the **ExtrinsicLSTM**  class. For sequential extrinsic type, ExtrinsicPOS function is implemented. For text classification same sequential model with overriding **def loadSamples(filename: String): Iterator[(String, String)]** and **override def labels(): Array[String]**. So basically if you want to use other datasets just modify the datasets. An example sequention and classification datasets are given as follows. 

### Sequential dataset

This dataset is a line by line tokenized dataset. Each line corresponds to a sample. Each label is separeted by **//** symbol and defined for every token.

```
Kimileri/NOUN buna/PRON kader/NOUN diyordu/VERB ,/PUNCT kimileri/NOUN unut/VERB ./PUNCT
Bu/PRON da/PART zaman/NOUN ister/VERB ,/PUNCT emek/NOUN ister/VERB ./PUNCT
İki/NUM veli/NOUN dokunulsa/VERB ağlayacaktı/_ ağlayacak/VERB tı/AUX ./PUNCT
```

### Classification dataset

This dataset is also a line by line dataset. Each line is pair of text and classification label. They are separeted by single tab (**\t**).

```
Neşe ve Üzüntü köprünün kırılmaya başlamasıyla geri dönerler .	Notr
i phone 5 ten sonra gene 4'' ekranı tercih ettim. telefon mükemmel. ertesi gün elime ulaştı.	Positive
Beşinci sezonda diziye yeni oyuncular katıldı .	Notr
```
The ExtrinsicNER class is given as follows. This class uses Self-Attention LSTM neural network model to test the accuracy of a sequential dataset. In order to modify a sequential task without changing the neural network model this class can be inherited with overriding all its methods. Note that the dataset must be in the correct form.

```scala
  
class ExtrinsicNER(params:SampleParams) extends ExtrinsicPOS(params){

  override def getClassifier(): String = "ner"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/ner/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/ner/test.txt"
  }

}
```

The ExtrinsicSentiment class is given as follows. This class uses ELMO neural network model and it changes the dataset loading methods. The loadSamples retrieves all the samples in the file. The content of the iterator is a sentence sample and label pairs. Each sample must be defined line by line as in the classification dataset. 

```scala
class ExtrinsicSentiment(params:SampleParams) extends ExtrinsicLSTM(params){

  var categories :Array[String] = null
  override def getClassifier(): String = "sentiment"

  override def getTraining(): String = {
    //dataset filename
    "resources/evaluation/sentiment/train.txt"
  }

  override def getTesing(): String = {
    //dataset filename
    "resources/evaluation/sentiment/test.txt"
  }

  override def loadSamples(filename: String): Iterator[(String, String)] = {

    Source.fromFile(filename).getLines().map(line=> {
      val Array(p1, p2) = line.split("\t")
      (p1, p2)
    })
  }

  override def labels(): Array[String] = {

    //predefined or extracted labels
    if(categories == null){
      categories = loadSamples(getTraining()).map(_._2).toSet.toArray
    }

    categories
  }
}

```

# Other Languages

There is no support for other languages but they can be implemented by applying custom tokenizer and a custom evaluation dataset. 

1. In order to support other languages first a frequent n-gram tokenizer must be implemented. The current tokenization is language independent but frequent n-grams are language dependent. The static vocabulary of an evaluation dataset must be constructed in selection stage.
2. A large line by line sentence dataset of the target language must be specified by a path in SampleParams. This text dataset will be used as a primary input in selection.
3. The tokenizer of the evaluation stage is only used in EmbeddingModel. This tokenizer instance can be modified in this class or a new class can be derived with an appropriate implementation of tokenize method.

Finally, with the correct folder paths defined in ExperimentSPL and SamplingExperiment classes a new language can be used. The paths may include the following

1. Training and testing datasets
2. The path to main sentence dataset
3. The name and folder of the target task
4. The vbocabulary file of the evaluation dataset
5. The implementation must be done for conditions for the target task in ExperimentSPL and SamplingExperiment. Currently, ner, pos, and sentiment are implemented.  

   
