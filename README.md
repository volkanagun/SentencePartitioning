# SentencePartitioning

The project is a maven project. The intellij is used as an IDE. It requires JDK11 and Scala 2.13.8 in order to compile. 

In this library, the effects of word partitioning is measured for the quality of the word embeddings extracted for POS tagging, NER, sentiment analysis and analogy tasks. Rather than giving the full parameteric details of the experimental setups, a short hand usage of the partitioning approach is presented in the following sections. In this respect two n-gram partitioning approaches is given below. 

* RankLM : It uses contextual ranking for partitioning word embeddings into useful n-grams
* SyllableLM: It partitions the words into valid syllables.


Both approaches produces multiple word splits separeted by \# symbol.


# Quick Details

The API provided here contains an [AbstractLM](https://github.com/volkanagun/SentencePartitioning/blob/master/src/transducer/AbstractLM.scala) class where the partitioning is trained and applied for new sentences through train and sentenceSplit methods. The training is done on a sentence corpus which is placed in resources/text/sentences folder. Sentence corpus is a line by line folder which also can be tokenized by space (\s) split. There is also a regex-based tokenizer provided to tokenize the sentences in the sentence corpus. To train the partitioning finite state model, the parameters must be defined. The parameters are determined in [Params](https://github.com/volkanagun/SentencePartitioning/blob/master/src/experiments/Params.scala). The language model definitions starts with "lm" prefix in Params variables. 

The most important parameters are 
* lmWindowLength: defines the token window size for inference.
* lmEpocs:  number of training epocs. When defined with lmMaxSentence, the total training sentence size becomes lmEopcs*lmMaxSentence.
* lmMaxSentence: number of training sentences in each epoc.
* lmTopSplit: number of candidate splits for each token. It defines the top most frequent or likely splits.

## Directory
![Directory folders](https://github.com/volkanagun/SentencePartitioning/blob/master/Directories.png?raw=true)

## Training

There are two improtant functions to be calledduring training. These are given in the following code segment.

```scala


## Inference
# Evaluation
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

There is no support for other languages but they can be implemented by applying custom tokenizer and a custom evaluation dataset. For the language support, the implementations of the target task must be specified in ExperimentSPL and SamplingExperiment. Currently, ner, pos, and sentiment are implemented. A checklist of necessary changes are given as follows. 

1. In order to support other languages first a frequent n-gram tokenizer must be implemented. The current tokenization is language independent but frequent n-grams are language dependent. The static vocabulary of an evaluation dataset must be constructed in selection stage.
2. A large line by line sentence dataset of the target language must be specified by a path in SampleParams. This text dataset will be used as a primary input in selection.
3. The tokenizer of the evaluation stage is only used in EmbeddingModel. This tokenizer instance can be modified in this class or a new class can be derived with an appropriate implementation of tokenize method.

Finally, with the correct folder paths defined in ExperimentSPL and SamplingExperiment classes a new language can be used. The paths may include the following

1. The path to main sentence dataset
2. Training and testing datasets
4. The name and folder of the target task
5. The vocabulary file of the evaluation dataset

Along with these changes the resources folder must include the path of these changes. A general overview of directory structure of the resources folder is given in the following screenshot.

![SamplingExperiment steps](https://github.com/volkanagun/ActiveSelection/blob/master/folders.png?raw=true)

   
