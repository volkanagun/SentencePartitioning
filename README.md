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
![Directory folders](https://github.com/volkanagun/SentencePartitioning/blob/master/directory.png?raw=true)

The folders above is the main folders and they are not totally required to run the project. 

* results: It stores the evaluation results for a selected task, embedding extraction method and n-gram partitioning profile.
* transducers: It stores the two level finite state machines for a specific task and n-gram partitioning profile.
* text/sentences: It contains the line by line sentence file. If you want to use your sentence file you must update the sentencesFile in Parameters.
* dictionary: It contains the lemma lexicon.txt file. If you have your own lemmatizer you must override the training in the [AbstractLM](https://github.com/volkanagun/SentencePartitioning/blob/master/src/transducer/AbstractLM.scala) class.
* binary: It contains meta files that contains the lemma dictionary, tokenizer specific files. These are not required for training but the existance of the folder is required.
* embeddings: It is the text or binary of the embedding files where they are stored as token:vector format.
* evaluation: It stores all the results in the experimental evaluation. So it is not required for quick runs.

## Training

There are two improtant functions to be called during training. These are given in the following code segment.

```scala
  def trainModel() : RankLM = {
    val parameters = createParams()
    val rankLM = new RankLM(parameters)
      .initialize()
      .loadTrain()
    
    rankLM.asInstanceOf[RankLM]

  }
```
Initialize function uses a weighted trie for training frequent sub-ngrams for tokens given by the lemma dictionary and text files. The dictionary  contains three filenames. These are given as follows.

```scala
class LMDictionary {

  val binaryFilename = "resources/binary/ranking.bin"
  var dictionaryTextFilename = "resources/dictionary/lexicon.txt"
  val sentenceFilename = "resources/text/sentences/sentences-tr.txt"
  val split = "#"
  var dictionaryTrie = new LMNode(0)

```
These files are used to initialize the lemma and other inflectional forms of the words in the text file. Parameters are used to limit the level of training because sentences-tr file can be huge. In order to train your own language words, the lemma dictionary as well as the initialization methods must be changed.

```scala
def fromDictionary(params: Params): LMDictionary = {

    if (params.lmTrainDictionary) {
      println("Constructing dictionary from text lexicon...")
      Source.fromFile(dictionaryTextFilename).getLines()
        .map(line => line.split("\t").head.toLowerCase(locale)
          .replaceAll("[\\~\\&\\_]", "")
          .replaceAll("([au])([bcçdfgğhjklmnprsştvyz])E$", "$1$2a")
          .replaceAll("([eü])([bcçdfgğhjklmnprsştvyz])E$", "$1$2e").toLowerCase(locale).trim)
        .foreach(token => {
          dictionaryTrie.add(Array(token, "END"))
        })

      println("Constructing dictionary from text lexicon finished")
    }

    this
  }
```

To change the fromDictionary training, you must obtain the lemmas of the words for the target language. These words must be added by 

```scala
dictionaryTrie.add(Array(word, "END")) 
```. 

## Inference

In the inference rankLM sentenceSplit method takes the tokenized sentences to predict the most likely sequence. It returns single sentence sequence which is partitioned into its n-grams.
 An example scala code is given below. In this example, parameter initialization, training and sentence inference is included

 ```scala
object ExampleLM {

  def createParams():Params={
    val parameters = new Params()

    parameters.adapterName = "rankLM"
    parameters.sentencesFile = "resources/text/sentences/sentences-tr.txt"
    parameters.lmWindowLength = 3
    parameters.lmEpocs = 1000
    parameters.lmMaxSentence = 240
    parameters.lmTopSplit = 3

    parameters
  }

  def trainModel() : RankLM={
    val parameters = createParams()
    val rankLM = new RankLM(parameters).initialize()
      .loadTrain()

    rankLM.asInstanceOf[RankLM]

  }

  def exampleSentence(): Unit = {

    val rankLM = trainModel()
    val sentences = Array[String](
      "Yaşamın ucuna yolculuk filmi gerçekten çok güzeldi .",
      "Saat ondan sonra burada bulunan alarm sistemi devreye giriyor .",
      "Başbakanın sözleri çok konuşuldu .",
      "Örneklerin elden geçmesi gerekiyordu ."
    )

    sentences.foreach(sentence=>{
      val tokens = sentence.split("\\s")
      val partitions = rankLM.splitSentence(tokens)
      val partitionString = partitions.mkString(" ")
      println(partitionString)
    })
  }

  def main(args: Array[String]): Unit = {
    exampleSentence()
  }
}

```


Inference of sentences for SyllableLM is also similar. Along with sentence splitting, token split can be obtained by the tokenSplit method of AbstractLM class. This methods creates candidate partitions for each token without using the context.

```scala
 def exampleToken(): Unit = {

    val rankLM = trainModel()
    val sentences = Array[String](
      "Yaşamın ucuna yolculuk filmi gerçekten çok güzeldi .",
      "Anıtkabire gelmek beni çok heyecanlandırdı ."
    )

    sentences.foreach(sentence=>{
      val tokens = sentence.split("\\s")
      tokens.foreach(token=>{
        rankLM.splitToken(token).foreach(split=>{
          println("Token: " + token + " Split: "+split)
        })
      })
    })
  }

```





