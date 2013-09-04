package com.mycompany.mia.classify

import scala.io.Source
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.mahout.vectorizer.encoders.{StaticWordValueEncoder, ContinuousValueEncoder}
import org.junit.Test
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.util.Version
import java.io.StringReader
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute

class FeatureHashingTest {

  @Test def testEncodeContinuousVariables() = {
    val encoder = new ContinuousValueEncoder("x")
    val vector = new RandomAccessSparseVector(40)
    val source = Source.fromFile("/tmp/donut.csv", "UTF-8")
    for (line <- source.getLines) {
      val x = line.split(",")(0)
      if (! "\"x\"".equals(x)) {
        encoder.addToVector(null, java.lang.Double.parseDouble(x), vector)
      } 
    }
    println(vector.toString)
  }
  
  @Test def testEncodeCategoricalOrWordLikeVariables() = {
    val encoder = new StaticWordValueEncoder("shape")
    val vector = new RandomAccessSparseVector(40)
    val source = Source.fromFile("/tmp/donut.csv", "UTF-8")
    for (line <- source.getLines) {
      val shape = line.split(",")(2)
      if (! "\"shape\"".equals(shape)) {
        encoder.addToVector(shape, vector)
      } 
    }
    println(vector.toString)
  }
  
  @Test def testEncodeTextLikeFeatures() = {
    val encoder = new StaticWordValueEncoder("text")
    val analyzer = new StandardAnalyzer(Version.LUCENE_43)
    val reader = new StringReader("text to magically vectorize")
    val tokstream = analyzer.tokenStream("body", reader)
    val termAttr = tokstream.addAttribute(classOf[CharTermAttribute])
    val vector = new RandomAccessSparseVector(100)
    while (tokstream.incrementToken()) {
      val word = new String(termAttr.buffer(), 0, termAttr.length())
      encoder.addToVector(word, 1, vector)
    }
    println(vector)
  }
}