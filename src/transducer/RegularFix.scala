package transducer

import scala.collection.mutable

class RegularFix {

  val replaceFix = Array(
    "([\\:\\.\\!\\?]+)([\"\'\\p{L}])", "$1 $2",
    "([\\:\\.\\!\\?]+)(\\s+?)\\d+$", "$1",
    "([\\”])", "\"",
    "([\\:\\.\\!\\?]+)(\\s+?)(\\[\\d+\\])","$3 $1"
  )

  val numberFix = Array(
    "(\\d+\\.\\d+((\\,\\d+)?))", "000.00",
    "(\\d+)", "000",
    "(\\d+\\.(\\d+))", "00.00",
    "(\\d+\\:(\\d+))", "00:00",
    "(\\d+\\.(\\d+)\\.\\d+)", "00.00.00",
  )



  def fixReplace(sentence: String): String = {
    var result = sentence
    replaceFix.sliding(2, 2).foreach { case Array(regex, replace) => {
      result = result.replaceAll(regex, replace)
    }}

    numberFix.sliding(2, 2).foreach { case Array(regex, replace) => {
      result = result.replaceAll(regex, replace)
    }}
    result
  }

  def fixDoubleQuoto(sentence: String): String = {
    sentence.replaceAll("\"(\\s+?)\"", "\"")
  }

  def fixDoubleSpace(sentence: String): String = {
    sentence.replaceAll("\\s+", " ")
  }

  def fixSentence(sentence: String): String = {
    val forwardQuoto = fixDoubleQuoto(fixQuoto(fixReplace(sentence)))
    val reverse = forwardQuoto.toCharArray.reverse.mkString
    val reverseQuoto = fixQuoto(reverse).toCharArray.reverse.mkString
    fixDoubleSpace(reverseQuoto)
  }

  def fixTokens(sentence:String):Array[String] = {
    sentence.split("\\s+").filter(token => !token.isEmpty)
  }

  def fixQuoto(sentence: String): String = {
    val characters = sentence.toCharArray.map(_.toString)
    var stack = mutable.Stack[String]()
    var crr = ""

    characters.foreach(character => {
      if (character.equals("\"")) {
        if (stack.isEmpty) {
          crr += character
          stack.push(character)
        }
        else crr += stack.pop()
      }
      else crr += character
    })

    if (!stack.isEmpty) crr += stack.pop()
    crr
  }

  def fixNums(sentence: String): String = {
    sentence.replaceAll("\\d", "0")
  }
}
