package com.mycompany.mia.cluster

import java.util.concurrent.atomic.AtomicInteger
import scala.collection.JavaConversions.{mapAsScalaMap, iterableAsScalaIterable, asScalaSet}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.IntWritable
import org.apache.hadoop.io.{Text, SequenceFile, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.apache.hadoop.mapreduce.{Mapper, Reducer, Job}
import org.apache.hadoop.util.GenericOptionsParser
import org.apache.mahout.math.{VectorWritable, SequentialAccessSparseVector, NamedVector}
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat

class DictionaryMapper 
    extends Mapper[LongWritable,Text,Text,Text] {
  
  val pattern = """<sep>""".r
  val zero = new IntWritable(0)
  
  override def map(key : LongWritable, 
      value : Text, 
      context : Mapper[LongWritable,Text,Text,Text]#Context) = { 
    val fields = pattern.split(value.toString)
    if (fields.length != 4) {
      context.getCounter("Map", "LinesWithErrors").increment(1)
    } else {
      val artist = fields(1)
      val tags = fields(2).toLowerCase().
      	replaceAll("""\p{Punct}""", " ").
      	replaceAll("""\s+""", " ").
      	split(" ")
      val count = Integer.parseInt(fields(3))
      for (tag <- tags) {
        context.write(new Text(artist), new Text(tag + ":" + count))
      }
    }
  }
}

class DictionaryReducer
    extends Reducer[Text,Text,Text,Text] {

  val zero = new IntWritable(0)

  override def reduce(key : Text, 
      values : java.lang.Iterable[Text],
      context : Reducer[Text,Text,Text,Text]#Context) = {
    val counts = new java.util.HashMap[String,Integer]
    for (value <- values) {
      val tagcount = value.toString.split(":")
      if (counts.containsKey(tagcount(0))) {
        val currCount = counts.get(tagcount(0))
        counts.put(tagcount(0), currCount + Integer.parseInt(tagcount(1)))
      } else {
        counts.put(tagcount(0), Integer.parseInt(tagcount(1)))
      }
    }
    val consolidated = counts.foldLeft("") {(xs, x) => xs + " " + x._1 + ":" + x._2}
    context.write(key, new Text(consolidated))
  }
}

class VectorMapper
    extends Mapper[Text,Text,IntWritable,VectorWritable] {
  
  val artistLookup = new java.util.HashMap[String,Integer]()
  val tagLookup = new java.util.HashMap[String,Integer]()
  
  override def setup(
      context : Mapper[Text,Text,IntWritable,VectorWritable]#Context) = {
    super.setup(context)
    val conf = context.getConfiguration()
    val apath = new Path(conf.get("artistFile"))
    val afs = FileSystem.get(apath.toUri(), conf)
    val artistReader = new SequenceFile.Reader(afs, apath, conf)
    val akey = new Text()
    val avalue = new IntWritable()
    while (artistReader.next(akey, avalue)) {
      artistLookup.put(akey.toString(), avalue.get())
    }
    artistReader.close()
    val tpath = new Path(conf.get("tagFile"))
    val tfs = FileSystem.get(tpath.toUri(), conf)
    val tagReader = new SequenceFile.Reader(tfs, tpath, conf)
    val tkey = new Text()
    val tvalue = new IntWritable()
    while (tagReader.next(tkey, tvalue)) {
      tagLookup.put(tkey.toString(), tvalue.get())
    }
    tagReader.close()
  }
  
  override def map(key : Text, value : Text,
      context : Mapper[Text,Text,IntWritable,VectorWritable]#Context) = {
    val artistId = artistLookup.get(key.toString)
    val vector = new NamedVector(
      new SequentialAccessSparseVector(tagLookup.size()), 
      key.toString)
    val tagCountPairs = value.toString.split(" ")
    for (tagCountPair <- tagCountPairs) {
      val tagcount = tagCountPair.split(":")
      if (tagcount.length == 2) {
        vector.set(tagLookup.get(tagcount(0)) - 1, 
          Integer.parseInt(tagcount(1)))
      }
    }
    context.write(new IntWritable(artistId), new VectorWritable(vector))
  }
}

class VectorReducer
    extends Reducer[IntWritable,VectorWritable,IntWritable,VectorWritable] {
  
  override def reduce(key : IntWritable,
      values : java.lang.Iterable[VectorWritable],
      context : Reducer[IntWritable,VectorWritable,IntWritable,VectorWritable]#Context) = {
    for (value <- values) {
      context.write(key, value)
    }
  }
}

/**
 * We need to read a file of the following format:
 * UUID<sep>Artist Name<sep>Tag<sep>Count
 * into a sequence file of [Text,VectorWritable] as follows:
 * Artist Name => VectorWritable(Tag:Count)
 */
object LastFMVectorizer {

  def main(args : Array[String]) : Int = {
    val conf = new Configuration()
    val otherArgs = (new GenericOptionsParser(conf, args)).getRemainingArgs
    if (otherArgs.length != 2) {
      println("Usage: LastFMVectorizer input_file output_dir")
      -1
    }
    // Dictionary Mapper/Reducer. Extract unique artists
    val job1 = new Job(conf, "Dictionary Mapper")
    job1.setJarByClass(classOf[DictionaryMapper])
    job1.setMapperClass(classOf[DictionaryMapper])
    job1.setReducerClass(classOf[DictionaryReducer])
    job1.setOutputKeyClass(classOf[Text])
    job1.setOutputValueClass(classOf[Text])
    job1.setInputFormatClass(classOf[TextInputFormat])
    job1.setOutputFormatClass(classOf[SequenceFileOutputFormat[Text,Text]])
    FileInputFormat.addInputPath(job1, new Path(args(0)))
    val dictOutput = new Path(args(1), "dictionary")
    FileOutputFormat.setOutputPath(job1, dictOutput)
    var succ = (job1.waitForCompletion(true))
    
    if (succ) {
      // get a mapping of unique ids to artist and tag for converting
      // to Mahout vectors
      val artistId = new AtomicInteger(0)
      val tagId = new AtomicInteger(0)
      val artistLookup = new java.util.HashMap[String,Integer]()
      val tagLookup = new java.util.HashMap[String,Integer]()
      val dictOutput = new Path(args(1), "dictionary")
      val fs = FileSystem.get(dictOutput.toUri(), conf)
      val outputfiles = fs.globStatus(new Path(dictOutput, "part-*"))
      var i = 0
      for (outputfile <- outputfiles) {
        val path = outputfile.getPath()
        val reader = new SequenceFile.Reader(fs, path, conf)
        val key = new Text()
        val value = new Text()
        while (reader.next(key, value)) {
          artistLookup.put(key.toString(), artistId.incrementAndGet())
          for (e <- value.toString.split(" ")) {
            val tagcount = e.split(":")
            if (! tagLookup.containsKey(tagcount(0))) {
              tagLookup.put(tagcount(0), tagId.incrementAndGet())
            }
          }
        }
      }
      // write them out into sequence files so they can be used
      // by the downstream Vectorizer job
      val artistWriter = new SequenceFile.Writer(
        fs, conf, new Path(args(1), "artists"), 
        classOf[Text], classOf[IntWritable])
      for (artist <- artistLookup.keySet()) {
        artistWriter.append(new Text(artist), 
          new IntWritable(artistLookup.get(artist)))
      }
      artistWriter.close()
      val tagWriter = new SequenceFile.Writer(
        fs, conf, new Path(args(1), "tags"),
        classOf[Text], classOf[IntWritable])
      for (tag <- tagLookup.keySet()) {
        tagWriter.append(new Text(tag), 
          new IntWritable(tagLookup.get(tag)))
      }
      tagWriter.close()
      
      conf.set("artistFile", args(1) + "/artists")
      conf.set("tagFile", args(1) + "/tags")
      val job2 = new Job(conf, "Dictionary Vectorizer")
      job2.setJarByClass(classOf[VectorMapper])
      job2.setMapperClass(classOf[VectorMapper])
      job2.setReducerClass(classOf[VectorReducer])
      job2.setOutputKeyClass(classOf[IntWritable])
      job2.setOutputValueClass(classOf[VectorWritable])
      job2.setInputFormatClass(classOf[SequenceFileInputFormat[Text,Text]])
      job2.setOutputFormatClass(classOf[SequenceFileOutputFormat[IntWritable,VectorWritable]])
      FileInputFormat.addInputPath(job2, new Path(args(1), "dictionary"))
      FileOutputFormat.setOutputPath(job2, new Path(args(1), "vectors"))
      succ = (job2.waitForCompletion(true))
    }
    if (succ) 0 else 1
  }
}

