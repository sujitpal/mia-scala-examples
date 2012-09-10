package com.mycompany.mia.classify

import java.io.{StringReader, PrintWriter, FileInputStream, File}
import java.util.{HashMap, Collections, ArrayList}

import scala.collection.JavaConversions.{collectionAsScalaIterable, asScalaBuffer}
import scala.io.Source

import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.lucene.util.Version
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression
import org.apache.mahout.classifier.sgd.{ModelSerializer, L1, AdaptiveLogisticRegression}
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.{Vector, RandomAccessSparseVector}
import org.apache.mahout.vectorizer.encoders.{TextValueEncoder, Dictionary, ConstantValueEncoder}

import com.google.common.collect.ConcurrentHashMultiset

object SGD20NewsgroupsClassifier extends App {

  val features = 10000
  val analyzer = new StandardAnalyzer(Version.LUCENE_32)
  val encoder = new TextValueEncoder("body")
  encoder.setProbes(2)
  val lines = new ConstantValueEncoder("line")
  val loglines = new ConstantValueEncoder("log(line)")
  val bias = new ConstantValueEncoder("intercept")
  val rand = RandomUtils.getRandom()

  // Usage: either
  // SGD20NewsgroupsClassifier train input_dir model_file dict_file, or
  // SGD20NewsgroupsClassifier test model_file dict_file test_dir
  args(0) match {
    case "train" => train(args(1), args(2), args(3))
    case "test" => test(args(1), args(2), args(3))
  }
  
  def train(trainDir : String, 
      modelFile : String, 
      dictFile : String) : Unit = {
    val newsgroups = new Dictionary()
    val learningAlgorithm = new AdaptiveLogisticRegression(
      20, features, new L1())
    learningAlgorithm.setInterval(800)
    learningAlgorithm.setAveragingWindow(500)
    // prepare data
    val files = new ArrayList[File]()
    val dirs = new File(trainDir).listFiles()
    for (dir <- dirs) {
      if (dir.isDirectory()) {
        newsgroups.intern(dir.getName())
        for (file <- dir.listFiles()) {
          files.add(file)
        }
      }
    }
    Collections.shuffle(files)
    println(files.size() + " training files in " + dirs.length + " classes")
  
    var k = 0
    var step = 0D
    var bumps = Array(1, 2, 5)

    for (file <- files) {
      val ng = file.getParentFile().getName()
      val actualClass = newsgroups.intern(ng)
      val vector = encodeFeatureVector(file)
      learningAlgorithm.train(actualClass, vector)
    
//      // optional: evaluating the model so far
//      k += 1
//      val bump = bumps(Math.floor(step).toInt % bumps.length)
//      val scale = Math.pow(10, Math.floor(step / bumps.length)).toInt
//      val best = learningAlgorithm.getBest()
//      var maxBeta = 0D
//      var nonZeros = 0D
//      var positive = 0D
//      var norm = 0D
//      var lambda = 0D
//      var mu = 0D
//      var averageLL = 0D
//      var averageCorrect = 0D
//    
//      if (best != null) {
//        val state = best.getPayload().getLearner()
//        averageCorrect = state.percentCorrect()
//        averageLL = state.logLikelihood()
//        val model = state.getModels().get(0)
//        // finish off pending regularization
//        model.close()
//      
//        val beta = model.getBeta()
//        maxBeta = beta.aggregate(Functions.MAX, Functions.ABS)
//        nonZeros = beta.aggregate(Functions.PLUS, new DoubleFunction() {
//          override def apply(v : Double) : Double = {
//            if (Math.abs(v) > 1.0e-6) 1 else 0
//          }
//        })
//        positive = beta.aggregate(Functions.PLUS, new DoubleFunction() {
//          override def apply(v : Double) : Double = {
//            if (v > 0) 1 else 0
//          }
//        })
//        norm = beta.aggregate(Functions.PLUS, Functions.ABS)
//        lambda = learningAlgorithm.getBest().getMappedParams()(0)
//        mu = learningAlgorithm.getBest().getMappedParams()(1)
//      } else {
//        maxBeta = 0
//        nonZeros = 0
//        positive = 0
//        norm = 0
//      }
//      if (k % (bump * scale) == 0) {
//        if (learningAlgorithm.getBest() != null) {
//          ModelSerializer.writeBinary("/tmp/newsgroup-" + k + " .model",
//            learningAlgorithm.getBest().getPayload().getLearner().getModels().get(0))
//        }
//        step += 0.25
//        printf("%.2f\t%.2f\t%.2f\t%.2f\t%.8g\t%.8g\t", 
//          maxBeta, nonZeros, positive, norm, lambda, mu)
//        printf("%d\t%.3f\t%.2f\n", k, averageLL, averageCorrect * 100)
//      }
//      // END optional: evaluating the model so far
    
    }
    learningAlgorithm.close()
//    dissect(newsgroups, learningAlgorithm, files)
    // evaluate model
    val learner = learningAlgorithm.getBest().getPayload().getLearner()
    println("AUC=" + learner.auc() + ", %-correct=" + learner.percentCorrect())
    ModelSerializer.writeBinary(modelFile, learner.getModels().get(0))
    val serializedDict = new PrintWriter(dictFile)
    for (newsgroup <- newsgroups.values()) {
      serializedDict.println(newsgroup)
    }
    serializedDict.flush()
    serializedDict.close()
  }
  
//  def dissect(newsgroups : Dictionary, 
//      learningAlgorithm : AdaptiveLogisticRegression, 
//      files : ArrayList[File]) : Unit = {
//    
//    val model = learningAlgorithm.getBest().getPayload().getLearner()
//    model.close()
//    
//    val traceDictionary = new java.util.HashMap[String,java.util.Set[Integer]]()
//    val modelDissector = new ModelDissector()
//    encoder.setTraceDictionary(traceDictionary)
//    bias.setTraceDictionary(traceDictionary)    
//    lines.setTraceDictionary(traceDictionary)
//    loglines.setTraceDictionary(traceDictionary)
//    for (file <- permute(files, rand).subList(0, 500)) {
//      traceDictionary.clear()
//      val vector = encodeFeatureVector(file)
//      modelDissector.update(vector, traceDictionary, model)
//    }
//    val ngnames = new ArrayList(newsgroups.values())
//    val weights = modelDissector.summary(100)
//    for (w <- weights) {
//      printf("%s\t%.1f\t%s\t%.1f\t%s\t%.1f\t%s\n",
//        w.getFeature(), w.getWeight(), ngnames.get(w.getMaxImpact() + 1),
//        w.getCategory(1), w.getWeight(1), w.getCategory(2), w.getWeight(2))
//    }
//  }
  
  def encodeFeatureVector(file : File) : Vector = {
    
    val vector = new RandomAccessSparseVector(features)
    val words : ConcurrentHashMultiset[String] = 
      ConcurrentHashMultiset.create()
    var numlines = 0
    var startBody = false
    var prevLine = ""
    for (line <- Source.fromFile(file).getLines()) {
      if (line.startsWith("From:") ||
          line.startsWith("Subject:") ||
          line.startsWith("Keywords:") ||
          line.startsWith("Summary:")) {
        countWords(line.replaceAll(".*:", ""), words)
      }
      if (! startBody &&
          line.trim().length() == 0 &&
          prevLine.trim().length() == 0) {
        startBody = true
      }
      if (startBody) {
        countWords(line, words)
      }
      numlines += 1
      prevLine = line
    }
    bias.addToVector(null, 1, vector)
    lines.addToVector(null, numlines / 30, vector)
    loglines.addToVector(null, Math.log(numlines + 1), vector)
    for (word <- words) {
      encoder.addToVector(word, Math.log(1 + words.count(word)), vector)
    }
    vector
  }
  
  def countWords(line : String,
      words : ConcurrentHashMultiset[String]) : Unit = {
    val words = new ArrayList[String]()
    val tokenStream = analyzer.tokenStream("text", new StringReader(line))
    tokenStream.addAttribute(classOf[CharTermAttribute])
    while (tokenStream.incrementToken()) {
      val attr = tokenStream.getAttribute(classOf[CharTermAttribute])
      words.add(new String(attr.buffer(), 0, attr.length()))
    }
  }
  
//  def permute(files : ArrayList[File], rand : Random) : ArrayList[File] = {
//    val permuted = new ArrayList[File]()
//    for (file <- files) {
//      val i = rand.nextInt(permuted.size() + 1)
//      if (i == permuted.size()) {
//        permuted.add(file)
//      } else {
//        permuted.add(permuted.get(i))
//        permuted.set(i, file)
//      }
//    }
//    permuted
//  }

  
  def test(modelFile : String, 
      dictFile : String, 
      testDir : String) : Unit = {
    
    val model = ModelSerializer.readBinary(
      new FileInputStream(modelFile), 
      classOf[OnlineLogisticRegression])
    val newsgroups = getNewsgroups(dictFile)
    
    val dirs = new File(testDir).listFiles()
    var ncorrect = 0
    var ntotal = 0
    for (dir <- dirs) {
      if (dir.isDirectory()) {
        val expectedLabel = dir.getName()
        for (file <- dir.listFiles()) {
          val vector = encodeFeatureVector(file)
          val results = model.classify(vector)
          val actualLabel = newsgroups.get(results.maxValueIndex())
          println("file: " + file.getName() + 
            ", expected: " + expectedLabel + 
            ", actual: " + actualLabel)
          if (actualLabel.equals(expectedLabel)) {
            ncorrect += 1
          }
          ntotal += 1
        }
      }
    }
    println("Correct: " + ncorrect + "/" + ntotal)
  }
  
  def getNewsgroups(dictFile : String) : HashMap[Integer,String] = {
    val newsgroups = new HashMap[Integer,String]()
    var lno = 0
    for (line <- Source.fromFile(dictFile).getLines()) {
      newsgroups.put(lno, line)
      lno += 1
    }
    newsgroups
  }
}