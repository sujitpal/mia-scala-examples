package com.mycompany.weka.autotagger

import java.io.PrintWriter
import java.io.FileWriter
import java.io.File
import scala.actors.threadpool.AtomicInteger
import scala.io.Source
import scala.math

object ArffGenerator extends App {

  // input files
  val XInput = "data/hlcms_X.txt"
  val yInput = "data/hlcms_y.txt"

  // output files
  val ArffOutput = "data/hlcms.arff"
  val Labels = "data/labels.txt"
  val Features = "data/mappings.txt"
    
  // set up label mappings, also write out map for future use
  val labels = Source.fromFile(new File(yInput)).
    getLines.
    toList
  val labelTuples = labels.toSet.zipWithIndex
  val labelToIdxMap = Map() ++ labelTuples
  val labelWriter = new PrintWriter(new FileWriter(new File(Labels)), true)
  labelTuples.foreach(t => labelWriter.println(t._1 + "\t" + t._2))
  labelWriter.flush()
  labelWriter.close()
  
  // set up feature mappings for future use
  val featureSet = Source.fromFile(new File(XInput)).
    getLines.
    map(line => {
      val pairs = line.split(" ")
      pairs.map(pair => pair.split("\\$")(0))
    }).
    flatten.
    toSet.
    zipWithIndex
  val featureToIdxMap = Map() ++ featureSet
  val featureWriter = new PrintWriter(new FileWriter(new File(Features)), true)
  featureSet.foreach(t => featureWriter.println(t._1 + "\t" + t._2))
  featureWriter.flush()
  featureWriter.close()

  // write out header information to ARFF file
  val arffWriter = new PrintWriter(new FileWriter(new File(ArffOutput)), true)
  arffWriter.println("@relation articlecms")
  arffWriter.println()
  
  featureSet.foreach(kv => 
    arffWriter.println("@attribute " + kv._1 + " numeric"))
  arffWriter.println("@attribute class {" + 
    labelTuples.map(kv => "'" + kv._2 + "'").mkString(",") + "}")
  arffWriter.println()
  
  arffWriter.println("@data")
  
  // read the X file, the y values are already set up in the
  // labels tuple above. L2 normalize scores per row, and 
  // write out a sparse ARFF file.
  var ln = 0
  val labelIndex = featureSet.size
  Source.fromFile(new File(XInput)).
    getLines.
    foreach(line => {
      Console.println(ln + ": " + line)
      val pairs = line.split(" ")
      val l2norm = scala.math.sqrt(pairs.
        map(pair => pair.split("\\$")(1).toDouble).
        foldLeft(0D)((a, b) => a + scala.math.pow(b, 2.0D)))
      val scoreTuples = pairs.map(pair => {
        val x = pair.split("\\$")
        (featureToIdxMap(x(0)), x(1).toDouble / l2norm)
        }).
        sortWith(_._1 < _._1).
        map(t => "%d %8.6f".format(t._1, t._2)).
        mkString(",")
      val label = labelToIdxMap(labels(ln))
      arffWriter.println("{%s,%d '%d'}".
        format(scoreTuples, labelIndex, label))
      ln += 1
  })
    
  arffWriter.flush()
  arffWriter.close()

}