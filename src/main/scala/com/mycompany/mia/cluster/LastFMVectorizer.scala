package com.mycompany.mia.cluster

import scala.collection.JavaConversions.iterableAsScalaIterable

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{IntWritable, DefaultStringifier, Text, SequenceFile, LongWritable}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, SequenceFileOutputFormat}
import org.apache.hadoop.mapreduce.{Mapper, Reducer, Job}
import org.apache.hadoop.util.{GenericsUtil, GenericOptionsParser}
import org.apache.mahout.math.{VectorWritable, Vector, SequentialAccessSparseVector, NamedVector}

/**
 * We need to read a file of the following format:
 * UUID<sep>Artist Name<sep>Tag<sep>Count
 * into a sequence file of [Text,VectorWritable] as follows:
 * Tag => VectorWritable(Artist:Count)
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
    job1.setOutputValueClass(classOf[IntWritable])
    job1.setInputFormatClass(classOf[TextInputFormat])
    job1.setOutputFormatClass(classOf[SequenceFileOutputFormat[Text,IntWritable]])
    FileInputFormat.addInputPath(job1, new Path(args(0)))
    val dictOutput = new Path(args(1), "dictionary")
    FileOutputFormat.setOutputPath(job1, dictOutput)
    var succ = (job1.waitForCompletion(true))
    
    if (succ) {
      // get a mapping of unique ids to artist and tag for converting
      // to Mahout vectors
      val dictOutput = new Path(args(1), "dictionary")
      val fs = FileSystem.get(dictOutput.toUri(), conf)
      val dictfiles = fs.globStatus(new Path(dictOutput, "part-*"))
      var i = 0
      val dictGlob = new Path(args(1), "dict-glob")
      val writer = new SequenceFile.Writer(fs, conf, dictGlob, 
        classOf[Text], classOf[IntWritable])
      for (dictfile <- dictfiles) {
        val path = dictfile.getPath()
        val reader = new SequenceFile.Reader(fs, path, conf)
        val key = new Text()
        val value = new IntWritable()
        while (reader.next(key, value)) {
          writer.append(key, new IntWritable(i))
          i += 1
        }
        reader.close()
      }
      writer.close()
      conf.set("dictpath", dictGlob.toString())

      val job2 = new Job(conf, "Dictionary Vectorizer")
      job2.setJarByClass(classOf[VectorMapper])
      job2.setMapperClass(classOf[VectorMapper])
      job2.setReducerClass(classOf[VectorReducer])
      job2.setOutputKeyClass(classOf[Text])
      job2.setOutputValueClass(classOf[VectorWritable])
      job2.setInputFormatClass(classOf[TextInputFormat])
      job2.setOutputFormatClass(classOf[SequenceFileOutputFormat[Text,VectorWritable]])
      FileInputFormat.addInputPath(job2, new Path(args(0)))
      FileOutputFormat.setOutputPath(job2, new Path(args(1), "vectors"))
      succ = (job2.waitForCompletion(true))
    }
    if (succ) 0 else 1
  }
}

/////////////////////////////////////////////////////////////////
// Assigns a unique ID to each artist. Needed by clusterdump
/////////////////////////////////////////////////////////////////

class DictionaryMapper 
    extends Mapper[LongWritable,Text,Text,IntWritable] {
  
  val pattern = """<sep>""".r
  val zero = new IntWritable(0)
  
  override def map(key : LongWritable, 
      value : Text, 
      context : Mapper[LongWritable,Text,Text,IntWritable]#Context) = { 
    val fields = pattern.split(value.toString)
    if (fields.length != 4) {
      context.getCounter("Map", "LinesWithErrors").increment(1)
    } else {
      context.write(new Text(fields(1)), zero)
    }
  }
}

class DictionaryReducer
    extends Reducer[Text,IntWritable,Text,IntWritable] {

  val zero = new IntWritable(0)

  override def reduce(key : Text, 
      values : java.lang.Iterable[IntWritable],
      context : Reducer[Text,IntWritable,Text,IntWritable]#Context) = {
    context.write(key, zero)
  }
}

/////////////////////////////////////////////////////////////////
// For each tag, creates feature vectors of artists
/////////////////////////////////////////////////////////////////

class VectorMapper
    extends Mapper[LongWritable,Text,Text,VectorWritable] {
  
  val pattern = """<sep>""".r
  var dict = new java.util.HashMap[String,Integer]()
  var vecwritable = new VectorWritable()
  
  override def setup(
      context : Mapper[LongWritable,Text,Text,VectorWritable]#Context) = {
    super.setup(context)
    val conf = context.getConfiguration()
    val dictpath = new Path(conf.get("dictpath"))
    val fs = FileSystem.get(dictpath.toUri(), conf)
    val reader = new SequenceFile.Reader(fs, dictpath, conf)
    val key = new Text()
    val value = new IntWritable()
    while (reader.next(key, value)) {
      dict.put(key.toString(), value.get())
    }
  }
  
  override def map(key : LongWritable, 
      value : Text,
      context : Mapper[LongWritable,Text,Text,VectorWritable]#Context) = {
    val fields = pattern.split(value.toString)
    if (fields.length != 4) {
      context.getCounter("Map", "LinesWithErrors").increment(1)
    } else {
      val artist = fields(1)
      val tag = fields(2)
      val weight = java.lang.Double.parseDouble(fields(3))
      val vector = new NamedVector(
        new SequentialAccessSparseVector(dict.size()), tag)
      vector.set(dict.get(artist), weight)
      vecwritable.set(vector)
      context.write(new Text(tag), vecwritable)
    }
  }
}

class VectorReducer
    extends Reducer[Text,VectorWritable,Text,VectorWritable] {
  
  var vecwritable = new VectorWritable()
  
  override def reduce(key : Text,
      values : java.lang.Iterable[VectorWritable],
      context : Reducer[Text,VectorWritable,Text,VectorWritable]#Context) = {
    var vector : Vector = null
    for (partialVector <- values) {
      if (vector == null) {
        vector = partialVector.get().like()
      } else {
        vector.plus(partialVector.get())
      }
    }
    val artistVector = new NamedVector(vector, key.toString())
    vecwritable.set(artistVector)
    context.write(key, vecwritable)
    for (value <- values) {
      context.write(key, value)
    }
  }
}

