package models

import experiments.Params
import glove.GloVe
import glove.objects.{Cooccurrence, Vocabulary, Word}
import glove.utils.Options
import org.jblas.DoubleMatrix
import transducer.AbstractLM
import utils.Tokenizer

import java.io.{FileOutputStream, ObjectOutputStream}
import java.util
import java.util.List

class GloveModel(params:Params, tokenizer:Tokenizer,  lm:AbstractLM) extends CBOWModel(params, tokenizer, lm) {

  var vectors:DoubleMatrix = null
  var vocab:Vocabulary = null


  override def train(filename: String): EmbeddingModel = {
    val options = new Options
    options.debug = true
    vocab = GloVe.build_vocabulary(filename, options)
    options.window_size = 3
    val c = GloVe.build_cooccurrence(vocab, filename, options)

    options.iterations = 10
    options.vector_size = 10
    options.debug = true
    vectors = GloVe.train(vocab, c, options)
    save()
  }

  override def save(): EmbeddingModel = {
    if(vectors!=null){
      println("Saving vector model...")
      val filename = params.embeddingsFilename()
      val words = vocab.iterate().toArray[Word](Array[Word]())
      val printer = new ObjectOutputStream(new FileOutputStream(filename))
      printer.writeInt(words.length)

      words.zipWithIndex.foreach {indPair => {
        val word = indPair._1
        val wordStr = word.getText
        val wordID = word.getId
        val wordVector = vectors.getColumn(wordID).data.map(_.toFloat)
        printer.writeObject(wordStr)
        printer.writeObject(wordVector)

        update(wordStr, wordVector)

      }}

      printer.close()
    }

    this
  }
}
