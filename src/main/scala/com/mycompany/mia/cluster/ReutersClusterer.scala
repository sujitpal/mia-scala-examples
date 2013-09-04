package com.mycompany.mia.cluster

import java.io.{StringReader, Reader}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{SequenceFile, IntWritable}
import org.apache.lucene.analysis.standard.{StandardTokenizer, StandardFilter, StandardAnalyzer}
import org.apache.lucene.util.Version
import org.apache.mahout.clustering.canopy.CanopyDriver
import org.apache.mahout.clustering.classify.WeightedVectorWritable
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansDriver
import org.apache.mahout.clustering.kmeans.KMeansDriver
import org.apache.mahout.clustering.Cluster
import org.apache.mahout.common.distance.{TanimotoDistanceMeasure, EuclideanDistanceMeasure}
import org.apache.mahout.common.HadoopUtil
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter
import org.apache.mahout.vectorizer.{DocumentProcessor, DictionaryVectorizer}
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.TokenStream
import org.apache.lucene.analysis.core.LowerCaseFilter
import org.apache.lucene.analysis.core.StopFilter
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.lucene.analysis.core.WhitespaceTokenizer
import org.apache.lucene.analysis.Analyzer.TokenStreamComponents
import org.apache.lucene.analysis.TokenFilter
import org.apache.lucene.analysis.Tokenizer

object ReutersClusterer extends App {
  
  // parameters
  val minSupport = 2
  val minDf = 5
  val maxDfPercent = 95
  val maxNGramSize = 2
  val minLLRValue = 50
  val reduceTasks = 1
  val chunkSize = 200
  val norm = 2
  val sequentialAccessOutput = true
  
  val inputDir = args(0) // directory of doc sequence file(s)
  val outputDir = args(1) // directory where clusters will be written
  val algo = args(2) // "kmeans" or "fkmeans"
  
  val conf = new Configuration()
  val fs = FileSystem.get(conf)
  HadoopUtil.delete(conf, new Path(outputDir))
  
  // converts input docs in sequence file format in input_dir 
  // into token array in output_dir/tokenized-documents
  val inputPath = new Path(inputDir)
  val tokenizedDocPath = new Path(outputDir, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER) 
  DocumentProcessor.tokenizeDocuments(inputPath, 
    classOf[ReutersAnalyzer], tokenizedDocPath, conf)
    
  // reads token array in output_dir/tokenized-documents and
  // writes term frequency vectors in output_dir (under tf-vectors)
  DictionaryVectorizer.createTermFrequencyVectors(
    tokenizedDocPath,
    new Path(outputDir),
    DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
    conf, minSupport, maxNGramSize, minLLRValue, 2, true, reduceTasks,
    chunkSize, sequentialAccessOutput, false)

  // converts term frequency vectors in output_dir/tf-vectors
  // to TF-IDF vectors in output_dir (under tfidf-vectors)
  val tfVectorPath = new Path(outputDir, DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER)
  val outputPath = new Path(outputDir)
  val docFreqs = TFIDFConverter.calculateDF(
    tfVectorPath, outputPath, conf, chunkSize)
  TFIDFConverter.processTfIdf(tfVectorPath, outputPath,
    conf, docFreqs, minDf, maxDfPercent, norm, true, 
    sequentialAccessOutput, false, reduceTasks)
    
  // reads tfidf-vectors from output_dir/tfidf-vectors
  // and writes out Canopy centroids at output_dir/canopy-centroids
  val tfidfVectorPath = new Path(outputDir, "tfidf-vectors")
  val canopyCentroidPath = new Path(outputDir, "canopy-centroids")
  CanopyDriver.run(conf, tfidfVectorPath, canopyCentroidPath,
    new EuclideanDistanceMeasure(), 250, 120, false, 0.01, false)
    
  // reads tfidf-vectors from output_dir/tfidf-vectors and 
  // refers to directory path for initial clusters, and 
  // writes out clusters to output_dir/clusters
  val clusterPath = new Path(outputDir, "clusters")
  algo match {
    case "kmeans" => KMeansDriver.run(conf, tfidfVectorPath, 
      new Path(canopyCentroidPath, "clusters-0-final"),
      clusterPath, new TanimotoDistanceMeasure(), 
      0.01, 20, true, 0.01, false)
    case "fkmeans" => FuzzyKMeansDriver.run(conf, tfidfVectorPath,
      new Path(canopyCentroidPath, "clusters-0-final"),
      clusterPath, new TanimotoDistanceMeasure(), 
      0.01, 20, 2.0f, true, true, 0.0, false)
    case _ => throw new IllegalArgumentException(
      "algo can be either kmeans or fkmeans")
  }
  
  // read clusters and output
  val reader = new SequenceFile.Reader(fs, 
    new Path(clusterPath, Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000"), 
    conf)
  val key = new IntWritable()
  val value = new WeightedVectorWritable()
  while (reader.next(key, value)) {
    println(key.toString + " belongs to " + value.toString)
  }
  reader.close()
}

class ReutersAnalyzer extends Analyzer {
  
  val ALPHA_PATTERN = """[a-z]+""".r
  
  override def createComponents(fieldName: String, reader: Reader):
      TokenStreamComponents = {
    // tokenize input string by standard tokenizer
    val source : Tokenizer = 
      new StandardTokenizer(Version.LUCENE_43, reader)
    var filter: TokenFilter = 
      new StandardFilter(Version.LUCENE_43, source)
    // lowercase all words
    filter = new LowerCaseFilter(Version.LUCENE_43, filter)
    // remove stop words
    filter = new StopFilter(Version.LUCENE_43, filter, 
      StandardAnalyzer.STOP_WORDS_SET)
    val termAttr = filter.addAttribute(classOf[CharTermAttribute]).
      asInstanceOf[CharTermAttribute]
    val buf = new StringBuilder()
    while (filter.incrementToken()) {
      // remove words < 3 chars long
      if (termAttr.length() >= 3) {
        val word = new String(
          termAttr.buffer(), 0, termAttr.length())
        // remove words with non-alpha chars in them
        if (ALPHA_PATTERN.pattern.matcher(word).matches) {
          buf.append(word).append(" ")
        }
      }
    }
    // return the remaining tokens
    new TokenStreamComponents(source, filter)
  }
}