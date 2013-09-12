package com.mycompany.mia.fpg

import java.io.{File, FileWriter, PrintWriter}
import scala.Array.canBuildFrom
import scala.collection.JavaConversions.asScalaBuffer
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, SequenceFile, Text}
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns
import scala.io.Source

object PostProcess extends App {

  val N = 12635756 // number of transactions (rows)
  val FreqFile = "data/fpg/fList"
  val FrequentPatternsDir = "data/fpg/frequentpatterns"
  val OutputFile = "data/imuid_rules.csv"

  val conf = new Configuration()
  val fs = FileSystem.get(conf)
  val frequency = computeFrequency(fs, conf, FreqFile)
  
  val writer = new PrintWriter(
    new FileWriter(new File(OutputFile)), true)

  val readers = new File(FrequentPatternsDir).list().
    filter(f => f.startsWith("part-r-")).
    map(f => new SequenceFile.Reader(fs, 
    new Path(FrequentPatternsDir, f), conf))
        
  readers.foreach(reader => {
    var key = new Text()
    var value = new TopKStringPatterns() // NOTE: deprecated in 0.8
    while (reader.next(key, value)) {
      val patterns = value.getPatterns()
      patterns.foreach(pattern => {
        // each pattern is a (concept_id_list,n) tuple that
        // states that the concepts in the list occurred n times.
        // - support for a pattern is given by n/N.
        // - each pattern translates to multiple rules, generated
        //   by rotating elements within a circular buffer, then
        //   making rules tail => head and calculate confidence
        //   for each rule as support / support(head).
        val items = pattern.getFirst()(0).split(" ")
        if (items.size > 1) {
          val support = (100.0D * pattern.getSecond()) / N
          items.foreach(item => {
            if (frequency.contains(item)) {
              val rest = items.filter(it => ! it.equals(item))
              val supportFirst = (100.0D * frequency(item)) / N
              val confidence = (100.0D * support) / supportFirst
              writer.println("""%5.3f,%5.3f,"%s => %s","%s => %s"""".format(
                support, confidence, rest.mkString("; "), item,
                rest.map(getName(_)).mkString("; "), getName(item)))
            }
          })
        }
      })
    }
    reader.close()
  })
  writer.flush()
  writer.close()
  
  def computeFrequency(fs: FileSystem,
      conf: Configuration, fList: String): Map[String,Long] = {
    val fs = FileSystem.get(conf)
    val reader = new SequenceFile.Reader(fs, new Path(fList), conf)
    var key = new Text()
    var value = new LongWritable()
    var freqs = Map[String,Long]()
    while (reader.next(key, value)) {
      freqs += ((key.toString(), value.get()))
    }
    reader.close()
    freqs
  }
  
  def getName(item: String): String = item
}
