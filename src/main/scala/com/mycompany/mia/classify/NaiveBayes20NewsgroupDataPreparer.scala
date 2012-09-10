package com.mycompany.mia.classify

import java.io.File

import scala.io.Source

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{Text, SequenceFile}

/**
 * Reads through the 2-level directory structure from the
 * 20-newsgroup dataset and writes out a sequence file in
 * the format (label, text), where label is the name of
 * the newsgroup (level 1) and text is the text of the file
 * (level 2). The output can be used by the Mahout seq2sparse
 * subcommand to create feature vectors.
 * 
 * Usage: sbt 'run-main \
 *   com.mycompany.mia.classify.NaiveBayes20NewsgroupDataPreparer \
 *   /path/to/20news-bydate-train \
 *   /path/to/20news-seq'
 */
object NaiveBayes20NewsgroupDataPreparer extends App {

  val conf = new Configuration()
  val fs = FileSystem.get(conf)
  val path = new Path(args(1))
  val writer = new SequenceFile.Writer(fs, conf, path, 
    classOf[Text], classOf[Text])
  val dirs = new File(args(0)).listFiles()
  var n = 0
  for (dir <- dirs) {
    val label = dir.getName()
    for (file <- dir.listFiles()) {
      val text = Source.fromFile(file).
        getLines().
        foldLeft("") (_ + " " + _)
      // extra slash added to key to get around AAOOB thrown
      // by BayesUtils.writeLabelIndex
      writer.append(new Text("/" + label), new Text(text))
      n += 1
    }
    println(label + ": " + n + " files loaded")
  }
  writer.close()
  
  // self-test to see that everything loaded okay...
  val reader = new SequenceFile.Reader(fs, path, conf)
  val key = new Text()
  val value = new Text()
  var rec = 0
  while (reader.next(key, value)) {
    if (rec < 10) {
      println(key.toString() + " => " + value.toString())
    }
    rec += 1
  }
  println("...")
  println("#=records written: " + rec)
  reader.close()
}
