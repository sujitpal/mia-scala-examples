package com.mycompany.weka.autotagger

import java.io.{File, FileWriter, PrintWriter}

import scala.io.Source

object AutoTagger extends App {

  val Xfile = new File("data/hlcms_X.txt")
  val Yfile = new File("data/hlcms_y.txt")
  val ModelFile = new File("data/model.txt")
  val TestFile = new File("data/hlcms_test.txt")
  val ReportFile = new File("data/hlcms_report.txt")
  
  val RecoPercentCutoff = 5
  val RecoNumber = 3
  
//  train(Xfile, Yfile, ModelFile)
  test(ModelFile, TestFile, ReportFile)
  
  def train(xfile: File, yfile: File, modelFile: File): Unit = {
    
    val matrices = scala.collection.mutable.Map[String,MapMatrix]()
  
    val yvalues = Source.fromFile(yfile).getLines.toList
    var ln = 0
    Source.fromFile(xfile).getLines.
      foreach(line => {
        val yvalue = yvalues(ln)
        val matrix = matrices.getOrElse(yvalue, new MapMatrix())
        matrix.addVector(MapVector.fromString(line, true))
        matrices(yvalue) = matrix
        ln = ln + 1    
    })
  
    val model = new PrintWriter(new FileWriter(modelFile), true)
    val x = matrices.keySet.
      map(key => (key, matrices(key).centroid())).
      foreach(pair => model.println("%s\t%s".
      format(pair._1, MapVector.toFormattedString(pair._2))))
    model.flush()
    model.close()
  }  
    
  def test(modelFile: File, testFile: File, reportFile: File): Unit = {
    
    val centroids = Source.fromFile(modelFile).getLines.
      map(line => {
        val cols = line.split("\t")
        (cols(0), MapVector.fromString(cols(1), true))
      }).
      toMap
    val writer = new PrintWriter(new FileWriter(reportFile), true)
    var numTests = 0.0D
    var numCorrect = 0.0D
    Source.fromFile(testFile).getLines.
      foreach(line => {
        val cols = line.split("\t")
        val catscores = centroids.keySet.map(key => {
            val vector = MapVector.fromString(cols(2), true)
            (key, vector.cosim(centroids(key)))
          }).
          toList.
          sortWith((a, b) => a._2 > b._2)
        val scoresum = catscores.map(kv => kv._2).
          foldLeft(0.0D)(_ + _)
        val confidences = catscores.map(kv => 
          (kv._1, kv._2 * 100 / scoresum)).
          filter(kv => kv._2 > RecoPercentCutoff).
          slice(0, RecoNumber)
        writer.println(cols(0)) // title
        writer.println("\t" + confidences.map(kv => 
          new String("%s (%-5.2f%%)".format(kv._1, kv._2))).
          mkString("; "))
        numTests = numTests + 1
        val recommended = confidences.map(_._1).toSet
        if (recommended.contains(cols(1))) 
          numCorrect = numCorrect + 1
    })
    writer.flush()
    writer.close()
    Console.println("Accuracy(%) = " + (numCorrect / numTests) * 100)
  }
  
}