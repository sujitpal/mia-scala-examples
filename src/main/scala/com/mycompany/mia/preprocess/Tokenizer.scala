package com.mycompany.mia.preprocess

import java.io.{File, FileInputStream, FileWriter, InputStream, PrintWriter}

import scala.Array.canBuildFrom
import scala.io.Source

import org.apache.commons.io.IOUtils

import opennlp.tools.chunker.{ChunkerME, ChunkerModel}
import opennlp.tools.postag.{POSModel, POSTaggerME}
import opennlp.tools.sentdetect.{SentenceDetectorME, SentenceModel}
import opennlp.tools.tokenize.{TokenizerME, TokenizerModel}

object Tokenizer extends App {
  val tokenizer = new Tokenizer()
  val idir = new File("/Users/sujit/Healthline/dev/people/spal/kaisertika/data/genentech/text")
  val writer = new PrintWriter(new FileWriter(
    new File("/Users/sujit/Healthline/dev/people/spal/kaisertika/data/genentech/phrases.txt")), true)
  idir.listFiles().foreach(file => {
    Source.fromFile(file).getLines().foreach(line => {
      val sentences = tokenizer.sentTokenize(line)
      sentences.foreach(sentence => {
        tokenizer.phraseChunk(sentence)
          .filter(phrase => "NP".equals(phrase._2))
          .foreach(phrase => writer.println(phrase._1 + "."))
      })
    })
  })
  writer.flush()
  writer.close()
}

class Tokenizer {

  val ModelDir = "/Users/sujit/Healthline/dev/people/spal/kaisertika/src/main/resources/models"

  val sentenceDetectorFn = (model: SentenceModel) =>
    new SentenceDetectorME(model)    
  val sentenceDetector = sentenceDetectorFn({
    var smis: InputStream = null
    try {
      smis = new FileInputStream(new File(ModelDir, "en_sent.bin"))
      val model = new SentenceModel(smis)
      model
    } finally {
      IOUtils.closeQuietly(smis)
    }   
  })
  val tokenizerFn = (model: TokenizerModel) => 
    new TokenizerME(model)
  val tokenizer = tokenizerFn({
    var tmis: InputStream = null
    try {
      tmis = new FileInputStream(new File(ModelDir, "en_token.bin"))
      val model = new TokenizerModel(tmis)
      model
    } finally {
      IOUtils.closeQuietly(tmis)
    }
  })
  val posTaggerFn = (model: POSModel) => 
    new POSTaggerME(model)
  val posTagger = posTaggerFn({
    var pmis: InputStream = null
    try {
      pmis = new FileInputStream(new File(ModelDir, "en_pos_maxent.bin"))
      val model = new POSModel(pmis)
      model
    } finally {
      IOUtils.closeQuietly(pmis)
    }
  })
  val chunkerFn = (model: ChunkerModel) => 
    new ChunkerME(model)
  val chunker = chunkerFn({
    var cmis: InputStream = null
    try {
      cmis = new FileInputStream(new File(ModelDir, "en_chunker.bin"))
      val model = new ChunkerModel(cmis)
      model
    } finally {
      IOUtils.closeQuietly(cmis)
    }
  })

  def sentTokenize(para: String): List[String] = {
    sentenceDetector.sentDetect(para).toList
  }
  
  def wordTokenize(sentence: String): List[String] = {
    return tokenizer.tokenize(sentence).toList
  }
  
  def posTag(sentence: String): List[(String,String)] = {
    val tokenSpans = tokenizer.tokenizePos(sentence)
    val tokens = tokenSpans.map(span => 
      span.getCoveredText(sentence).toString())
    val tags = posTagger.tag(tokens)
    tokens.zip(tags).toList
  }
  
  def phraseChunk(sentence: String): List[(String,String)] = {
    val tokenSpans = tokenizer.tokenizePos(sentence)
    val tokens = tokenSpans.map(span => 
      span.getCoveredText(sentence).toString())
    val tags = posTagger.tag(tokens)
    return chunker.chunkAsSpans(tokens, tags).map(chunk => {
      val start = tokenSpans(chunk.getStart()).getStart()
      val end = tokenSpans(chunk.getEnd() - 1).getEnd()
      (sentence.substring(start, end), chunk.getType())
    }).toList
  }
}
