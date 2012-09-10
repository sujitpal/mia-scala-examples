package com.mycompany.mia.classify

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.io.SequenceFile
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.Text
import org.apache.hadoop.io.IntWritable
import org.apache.mahout.math.VectorWritable
import org.apache.hadoop.io.LongWritable
import org.apache.mahout.common.StringTuple

/**
 * Crude but useful tool to look at the files created by seq2sparse.
 * Following files are created:
 * * df-count/part-r-00000 - SEQ[IntWritabke,LongWritable] - map of
 *   {termId => numdocs}
 * * dictionary.file-0 - SEQ[Text,IntWritable] - map of {term => termId}
 * * tf-vectors/part-r-00000 - SEQ[Text,VectorWritable] - map of 
 *   {label => document tf-vector}
 * * tf-idf-vectors/part-r-00000 - SEQ[Text,VectorWritable] - map of
 *   {label => document's tf/idf vector}
 * * tokenized-documents/part-r-00000 - SEQ[Text,StringTuple] - map 
 *   of {label => array of words after tokenization}.
 * * wordcount/part-r-00000 - SEQ[Text,LongWritbale] - map of {term, count}.
 * 
 * It can be run as follows:
 * sbt 'run-main com.mycompany.mia.classify \
 *   /non/HDFS/path/to/file key-class-alias value-class-alias'
 */
object SequenceFileReader extends App {

  val conf = new Configuration()
  val fs = FileSystem.get(conf)
  val reader = new SequenceFile.Reader(fs, new Path(args(0)), conf)
  val key = args(1) match {
    case "text" => new Text()
    case "int" => new IntWritable()
  } 
  val value = args(2) match {
    case "text" => new Text()
    case "int" => new IntWritable()
    case "long" => new LongWritable()
    case "string" => new StringTuple()
    case "vector" => new VectorWritable()
  }
  var i = 0
  while (reader.next(key, value)) {
    if (key != null && value != null) {
      if (i < 10) {
        val keystr = args(1) match {
          case "text" => key.toString()
          case "int" => key.asInstanceOf[IntWritable].get()
        } 
        val valuestr = args(2) match {
          case "text" => value.toString()
          case "int" => value.asInstanceOf[IntWritable].get()
          case "long" => value.asInstanceOf[LongWritable].get()
          case "string" => value.asInstanceOf[StringTuple].toString()
          case "vector" => value.asInstanceOf[VectorWritable].get()
        }
        println(keystr + " => " + valuestr)
      }
    }
    i += 1
  }
  reader.close()
  println("#-records: " + i)
}