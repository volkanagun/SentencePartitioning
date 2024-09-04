package transducer

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.collection.parallel.CollectionConverters.ArrayIsParallelizable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.{Await, Future, Promise, TimeoutException}
import scala.concurrent.duration.Duration
import scala.util.matching.Regex
import scala.util.{Failure, Success}

/**
 * @Author Dr. Hayri Volkan Agun
 * @Date 20.03.2022 17:29
 * @Project BigLanguage
 * @Version 1.0
 */

class TransducerLM(var transducer: Transducer, var seqTransducer: Transducer = new Transducer()) extends Serializable {

  def copy(): TransducerLM = {
    new TransducerLM(transducer.copy(), seqTransducer.copy())
  }

  def copyBase(): TransducerLM = {
    new TransducerLM(transducer.copy(), new Transducer())
  }

  def isEmpty(): Boolean = {
    transducer.map.isEmpty || seqTransducer.map.isEmpty
  }

  def graphStats():Map[String, Double] = {
    seqTransducer.stats()
  }

  def merge(transducerLM: TransducerLM): TransducerLM = {
    println("Merging transducers")
    synchronized {
      transducer.merge(transducerLM.transducer)
      seqTransducer.merge(transducerLM.seqTransducer)
    }
    this
  }

  def mergeSequence(transducerLM: TransducerLM): TransducerLM = {
    println("Merging seq transducers")
    synchronized {
      seqTransducer.merge(transducerLM.seqTransducer)
    }
    this
  }

  def prune(n: Int = 100): TransducerLM = {
    println("Prunning transducers")
    seqTransducer.prune(n)
    this
  }

  def normalize(): TransducerLM = {
    println("Normalizing transducers")
    transducer.normalize()
    seqTransducer.normalize()
    this
  }

  def setTransducer(transducer: Transducer): this.type = {
    this.transducer = transducer
    this
  }


  def combinatoric(input: Array[Array[String]], result: Array[Array[String]] = Array[Array[String]](Array()), i: Int = 0): Array[Array[String]] = {

    if (i == input.length) result
    else {

      var crr = i;
      val dist = input(crr).distinct

      var array = Array[Array[String]]()
      for (k <- 0 until dist.length) {
        for (j <- 0 until result.length) {
          val current = result(j) :+ dist(k)
          array = array :+ current
        }
      }

      combinatoric(input, array, crr + 1)
    }
  }


  def countCombinatoric(sequence: Array[String], top: Int, slide: Int): Unit = {

    val combinationSpace = combinatoric(sequence.map(token => transducer.multipleSplitSearch(token, top)))
    combinationSpace.foreach(sequence => {
      sequence.flatMap(item => item.split(transducer.split)).sliding(slide, 1)
        .toArray.foreach(subitems => {
          seqTransducer.addPrefix(subitems)
        })
    })
  }

  def countEfficient(sequence: Array[String], sample: Int, top: Int, slide: Int, skip: Int): Unit = {
    sequence.map(token => transducer.multipleEfficientSearch(token, top, sample))
      .sliding(slide, 1).foreach(crrCombinations => {
         seqTransducer.addSkipEfficient(crrCombinations, skip)

      })
  }


  def countCombinatoric(sequence: Array[String], sample: Int, top: Int, slide: Int, skip: Int): Unit = {
    sequence.map(token => transducer.multipleSplitSearch(token, top, sample))
      .sliding(slide, 1).foreach(crrCombinations => {
        seqTransducer.addSkipEfficient(crrCombinations, skip)
      })
  }


  def countCombinatoric(sequence: Array[String], top: Int, slide: Int, skip: Int): Unit = {

    val combinationSpace = combinatoric(sequence.map(token => transducer.multipleSplitSearch(token, top)))
    combinationSpace.foreach(sequence => {
      sequence.flatMap(item => item.split(transducer.split)).sliding(slide, 1)
        .toArray.foreach(subitems => {
          seqTransducer.addSkip(subitems, skip)
        })
    })
  }

  def countEfficientCombinatoric(sequence: Array[String], top: Int, slide: Int, skip: Int): Unit = {

    val combinationSpace = combinatoric(sequence.map(token => transducer.multipleEfficientSearch(token, top)))
    combinationSpace.foreach(sequence => {
      sequence.flatMap(item => item.split(transducer.split)).sliding(slide, 1)
        .toArray.foreach(subitems => {
          seqTransducer.addSkip(subitems, skip)
        })
    })
  }

  def countCombinatoric(sequence: Array[String], partitionFunc: (Array[String]) => Array[Array[String]], slide: Int, skip: Int): Unit = {

    val combinationSpace = combinatoric(partitionFunc(sequence))
    combinationSpace.foreach(sequence => {
      sequence.flatMap(item => item.split(transducer.split)).sliding(slide, 1)
        .toArray.foreach(subitems => {
          seqTransducer.addSkip(subitems, skip)
        })
    })
  }

  def countEfficientCombinatoric(sequence: Array[String], partitionFunc: (Array[String]) => Array[Array[String]], sample:Int, slide: Int, skip: Int): Unit = {

    val combinationSpace = partitionFunc(sequence)
    combinationSpace.sliding(slide, 1).foreach(crrCombinations => {
      seqTransducer.addSkipEfficient(crrCombinations, skip)
    })
  }


  def count(sequence: String, top: Int, slide: Int): Unit = {
    transducer.multipleSplitSearch(sequence, top)
      .foreach(item => {
        item.split(transducer.marker).sliding(slide, 1)
          .toArray.foreach(subitems => {
            seqTransducer.addPrefix(subitems)
          })
      })
  }

  def count(sequence: Array[String], top: Int, slide: Int): Unit = {

    sequence.flatMap(token => {
        transducer.multipleSplitSearch(token, top)
      })
      .foreach(item => {
        item.split(transducer.marker).sliding(slide, 1)
          .toArray.foreach(subitems => {
            seqTransducer.addPrefix(subitems)
          })
      })
  }

  def count(sequence: Array[String], top: Int, slide: Int, skip: Int): Unit = {

    sequence.flatMap(token => {
        transducer.multipleSplitSearch(token, top)
      })
      .foreach(item => {
        item.split(transducer.marker).sliding(slide, 1)
          .toArray.foreach(subitems => {
            seqTransducer.addSkip(subitems, skip)
          })
      })
  }

  def count(sequence: String, top: Int, slide: Int, skip: Int): Unit = {
    transducer.multipleSplitSearch(sequence, top)
      .foreach(item => {
        item.split(transducer.marker).sliding(slide, 1)
          .toArray.foreach(subitems => {
            seqTransducer.addSkip(subitems, skip)
          })
      })
  }

  def infer(sequence: String, top: Int): Array[String] = {
    transducer.multipleSplitSearch(sequence, top).map(item => {
        val slice = item.split(transducer.marker)
        (item, seqTransducer.likelyhoodSearch(slice))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def infer(sequence: Array[String]): Array[String] = {
    transducer.multipleSearch(sequence).map(item => {
        val slice = item.sequence.split(transducer.marker)
        (item.sequence, seqTransducer.likelyhoodSearch(slice))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def skipLink(sequence: Array[String], skip: Int, top: Int): Array[String] = {
    val input = sequence.map(item => transducer.multipleSplitSearch(item, top))
    val combinations = combinatoric(input)

    combinations.map(slice => {
        val split = slice.flatMap(item => item.split(transducer.split))
        (slice.mkString(" "), seqTransducer.skipSearch(split, skip))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def skipEfficientLink(sequence: Array[String], skip: Int, top: Int): Array[String] = {
    val input = sequence.map(item => transducer.multipleEfficientSearch(item, top))
    val combinations = combinatoric(input)

    combinations.map(slice => {
        val split = slice.flatMap(item => item.split(transducer.split))
        (slice.mkString(" "), seqTransducer.skipSearch(split, skip))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def skipSlideLink(sequence: Array[String], window: Int, skip: Int, top: Int): Array[String] = {
    val partitions = sequence.sliding(window, 1).map(windowSeq => {
      val input = windowSeq.map(item => transducer.multipleSplitSearch(item, top))
      val combinations = combinatoric(input)

      combinations.map(slice => {
          val split = slice.flatMap(item => item.split(transducer.split))
          (slice, seqTransducer.skipSearch(split, skip))
        }).sortBy(item => item._2)
        .reverse
        .map(_._1).head
    }).toArray

    var tokens = partitions.head
    partitions.tail.foreach(partition => {
      tokens :+= partition.last
    })

    tokens
  }

  def skipLink(sequence: String, skip: Int, top: Int): Array[String] = {
    transducer.multipleSplitSearch(sequence, top).map(item => {
        val slice = item.split(transducer.marker)
        (item, seqTransducer.skipSearch(slice, skip))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def pageRank(sequence: Array[String], skip: Int, top: Int, iter: Int): Array[String] = {
    val input = sequence.map(item => transducer.multipleSplitSearch(item, top))
    val combinations = combinatoric(input)

    combinations.map(combinationSequence => {
        val sliceSplit = combinationSequence.flatMap(item => item.split(transducer.split))
        (combinationSequence.mkString(" "), seqTransducer.rankingSearch(sliceSplit, skip, iter))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def pageRankEfficient(sequence: Array[String], sample: Int, skip: Int, top: Int, iter: Int): Array[String] = {
    val input = sequence.map(item => transducer.multipleEfficientSearch(item, top, sample))
    val combinations = combinatoric(input)

    combinations.map(combinationSequence => {
        val sliceSplit = combinationSequence.flatMap(item => item.split(transducer.split))
        (combinationSequence.mkString(" "), seqTransducer.rankingSearch(sliceSplit, skip, iter))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def pageSlideRank(sequence: Array[String], windowLength: Int, skip: Int, top: Int, iter: Int): Array[String] = {
    val partitions = sequence.sliding(windowLength, 1).map(seq => {
      val input = seq.map(item => transducer.multipleSplitSearch(item, top))

      val combinations = combinatoric(input)

      combinations.map(combinationSequence => {
          val sliceSplit = combinationSequence.flatMap(item => item.split(transducer.split))
          (combinationSequence, seqTransducer.rankingSearch(sliceSplit, skip, iter))
        }).sortBy(item => item._2)
        .reverse
        .map(_._1).head
    }).toArray

    var tokens = partitions.head
    partitions.tail.foreach(item => {
      tokens = tokens ++ item.last.split(transducer.split)
    })

    tokens

  }

  def pageSlideRank(sequence: Array[String], partitionFunc: (Array[String] => Array[Array[String]]), slideLength: Int, skip: Int, top: Int, iter: Int): Array[String] = {

    val partitions = sequence.sliding(slideLength, 1).zipWithIndex.toArray.par.map {
        case (seqwindow, iwindow) => {
          val partitionTokens = {
            val input = partitionFunc(seqwindow)
            val combinations = combinatoric(input)
            val scored = combinations.par.map(slice => {
                val splitted = slice.flatMap(item => item.split(transducer.split))
                (slice, seqTransducer.rankingSearch(splitted, skip, iter))
              }).toArray.sortBy(item => item._2)
              .reverse

            scored.map(_._1).head
          }

          (partitionTokens, iwindow)
        }
      }.toArray.sortBy(_._2)
      .map(_._1)

    var tokens = partitions.head
    partitions.tail.foreach(items => {
      tokens :+= items.last
    })

    tokens
  }

  def pageRank(sequence: Array[String], partitionFunc: (Array[String] => Array[Array[String]]), skip: Int, top: Int, iter: Int): Array[String] = {
    val input = partitionFunc(sequence)
    val combinations = combinatoric(input)
    val scored = combinations.map(slice => {
        val splitted = slice.flatMap(item => item.split(transducer.split))
        (slice.mkString(" "), seqTransducer.rankingSearch(splitted, skip, iter))
      }).sortBy(item => item._2)
      .reverse

    scored.map(_._1)
  }


  def suffixRank(sequence: Array[String], skip: Int, top: Int, iter: Int): Array[String] = {
    val input = sequence.map(item => transducer.suffixSplitSearch(item, top))
    val combinations = combinatoric(input)

    combinations.map(slice => {
        val splitted = slice.flatMap(item => item.split(transducer.split))
        (slice.mkString(" "), seqTransducer.rankingSearch(splitted, skip, iter))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def pageRank(sequence: String, skip: Int, top: Int, iter: Int): Array[String] = {
    transducer.multipleSplitSearch(sequence, top).map(item => {
        val slice = item.split(transducer.marker)
        (item, seqTransducer.rankingSearch(slice, skip, iter))
      }).sortBy(item => item._2)
      .reverse
      .map(_._1)
  }

  def inferMinSplit(sequence: String, top: Int): Array[String] = {
    val space = "(\\s?)" + transducer.marker + "(\\s?)"
    val results = transducer.multipleSplitSearch(sequence, top).map(item => {
        val slice = item.split(space).map(_.trim).filter(!_.isEmpty)
        (item, seqTransducer.likelyhoodSearch(slice) * slice.length)
      }).groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .map(_._1)

    results
  }

  def inferMinSplit(sequence: String, top: Int, accept: (String) => Boolean): Array[String] = {
    val space = "(\\s?)" + transducer.marker + "(\\s?)"
    val results = transducer.multipleSplitSearch(sequence, top)
      .filter(item => accept(item))
      .map(item => {
        val slice = item.split(space).map(_.trim).filter(!_.isEmpty)
        (item, seqTransducer.likelyhoodSearch(slice) * slice.length)
      }).groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .map(_._1)
    results
  }

  def inferMinTokenSplit(sequence: Array[String], top: Int): Array[String] = {
    val input = sequence.map(item => transducer.multipleSplitSearch(item, top))

    val combinations = combinatoric(input)

    val results = combinations
      .par
      .map(items => {
        val slice = items.flatMap(item => item.split(transducer.split).map(_.trim).filter(!_.isEmpty))
        (items.mkString(" "), seqTransducer.likelyhoodSearch(slice) * slice.length)
      }).toArray.groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .map(_._1)


    results
  }

  def inferMinTokenSplit(sequence: Array[String], notAccept: Array[Regex], top: Int, topCombination: Int): Array[String] = {
    var input = sequence.map(item => transducer.multipleSplitSearch(item, top))
    val inputFilter = input.map(sequence => sequence.filter(item => !notAccept.exists(r => r.findFirstIn(item).nonEmpty)).take(topCombination))
      .filter(_.nonEmpty)

    val combinations = combinatoric(inputFilter)

    val results = combinations
      .par
      .map(items => {
        val slice = items.flatMap(item => item.split(transducer.split).map(_.trim).filter(!_.isEmpty))
        (items.mkString(" "), seqTransducer.likelyhoodSearch(slice) * slice.length)
      }).toArray.groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .map(_._1)


    results
  }

  def inferMinEfficientSplit(sequence: Array[String], notAccept: Array[Regex], top: Int, topCombination: Int): Array[String] = {
    var input = sequence.map(item => transducer.multipleEfficientSearch(item, top))
    val inputFilter = input.map(sequence => sequence.filter(item => !notAccept.exists(r => r.findFirstIn(item).nonEmpty)).take(topCombination))
      .filter(_.nonEmpty)

    val combinations = combinatoric(inputFilter)

    val results = combinations
      .par
      .map(items => {
        val slice = items.flatMap(item => item.split(transducer.split).map(_.trim).filter(!_.isEmpty))
        (items.mkString(" "), seqTransducer.likelyhoodSearch(slice) * slice.length)
      }).toArray.groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .map(_._1)


    results
  }

  def inferSlideMinTokenSplit(sequence: Array[String], windowLength: Int, top: Int): Array[String] = {

    val partitions = sequence.sliding(windowLength, 1).map(windowSeq => {
      val input = windowSeq.map(item => transducer.multipleSplitSearch(item, top))
      val combinations = combinatoric(input)

      val results = combinations
        .par
        .map(items => {
          val slice = items.flatMap(item => item.split(transducer.split).map(_.trim).filter(!_.isEmpty))
          (items.mkString(" "), seqTransducer.likelyhoodSearch(slice) * slice.length)
        }).toArray.groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
        .toArray.sortBy(item => item._2)
        .reverse
        .map(_._1)
        .head.split("\\s")

      results
    }).toArray

    var tokens = partitions.head
    partitions.tail.foreach(items => tokens :+= items.last)
    tokens
  }

  def inferMultiTokenSplit(sequence: Array[String], top: Int): Array[String] = {
    val input = sequence.map(item => transducer.multipleSplitSearch(item, top))
    val combinations = combinatoric(input)

    val results = combinations
      .map(items => {
        val slice = items.flatMap(item => item.split(transducer.split).map(_.trim).filter(!_.isEmpty))
        (items.mkString(" "), seqTransducer.likelyhoodSearch(slice) * slice.length)
      }).groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .take(top)
      .map(_._1)

    val splitted = results.map(item => item.split(transducer.split).mkString(" "))
    results ++ splitted
  }

  def inferLikelihood(sequence: Array[String], top: Int): Array[String] = {
    val space = "(\\s?)" + transducer.marker + "(\\s?)"
    val input = sequence.map(item => transducer.multipleSplitSearch(item, top))
    val combinations = combinatoric(input)

    val results = combinations
      .map(items => {
        val slice = items.flatMap(item => item.split(space).map(_.trim).filter(!_.isEmpty))
        (items.mkString(" "), seqTransducer.likelyhoodSearch(slice))
      }).groupBy(_._1).view.mapValues(items => items.map(_._2).sum)
      .toArray.sortBy(item => item._2)
      .reverse
      .map(_._1)
      .head
      .split("\\s")

    results
  }


  def inferSplitScores(sequence: Array[String]): Array[(String, Double)] = {
    transducer.multipleSearch(sequence).map(item => {
        val slice = item.sequence.split(transducer.marker)
        (item.sequence, seqTransducer.likelyhoodSearch(slice))
      }).sortBy(item => item._2)
      .reverse
  }

  def inferLogSum(token: String, top: Int): Array[(String, Double)] = {
    transducer.multipleSplitSearch(token, top).map(item => {
        val slice = item.split(transducer.marker)
        (item, seqTransducer.likelyhoodSearch(slice))
      }).sortBy(item => item._2)
      .reverse
  }

  def save(out: ObjectOutputStream): Unit = {
    transducer.save(out)
    seqTransducer.save(out)
  }

  def load(in: ObjectInputStream): Unit = {
    transducer.load(in)
    seqTransducer.load(in)
  }

  def test(): Unit = {
    println("Dictionary finished states empty: " + transducer.finished.isEmpty)
    println("Dictionary map empty: " + transducer.map.isEmpty)
    if (transducer.map.contains(0)) println("Dictionary map(0) empty: " + transducer.map(0L).stateMap.isEmpty)
    else println("Dictionary map(0) empty: true")

    println("Finished states empty: " + seqTransducer.finished.isEmpty)
    println("Map empty: " + seqTransducer.map.isEmpty)
    //transducer.test()
    //inferMinSplit("akdenizdekilerden", 10)
    //inferMinSplit("akdenizdekiler", 10)
    //inferMinSplit("akdenizdeki", 10)

  }

}
