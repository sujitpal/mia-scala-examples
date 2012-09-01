package com.mycompany.mia.cluster

import java.util.ArrayList
import scala.collection.JavaConversions.seqAsJavaList
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{Text, SequenceFile, LongWritable, IntWritable}
import org.apache.mahout.clustering.classify.WeightedVectorWritable
import org.apache.mahout.clustering.kmeans.KMeansDriver
import org.apache.mahout.clustering.Cluster
import org.apache.mahout.common.distance.EuclideanDistanceMeasure
import org.apache.mahout.math.{VectorWritable, Vector, RandomAccessSparseVector}
import org.junit.Test
import org.apache.mahout.clustering.iterator.DistanceMeasureCluster
import org.apache.mahout.clustering.kmeans.Kluster
import scala.collection.mutable.ArrayBuffer
import org.apache.mahout.math.NamedVector
import org.apache.mahout.math.DenseVector

class ClusterTests {

  // mkdir -p data/clustering/clusters, input, output
  @Test def testClusterPoints() = {
    
    val points : Array[(Double,Double)] = Array(
      (1, 1), (2, 1), (1, 2), 
      (2, 2), (3, 3), (8, 8),
      (9, 8), (8, 9), (9, 9))
    val k = 2
  
    val vectors = getPoints(points)
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    writePointsToFile(vectors, "data/clustering/input/file1", fs, conf)
    val path = new Path("data/clustering/clusters/part-00000")
    val writer = new SequenceFile.Writer(fs, conf, path, 
      classOf[Text], classOf[Kluster])
    for (i <- 0 to k) {
      val vec = vectors.get(i)
      val cluster = new Kluster(vec, i, new EuclideanDistanceMeasure())
      writer.append(new Text(cluster.getIdentifier()), cluster)
    }
    writer.close()
    KMeansDriver.run(conf, 
      new Path("data/clustering/input"), 
      new Path("data/clustering/clusters"),
      new Path("data/clustering/output"),
      new EuclideanDistanceMeasure(), 
      0.001, 10, true, 0.5, false)
    val reader = new SequenceFile.Reader(fs, 
      new Path("data/clustering/output/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000"), 
      conf)
    val key = new IntWritable()
    val value = new WeightedVectorWritable()
    while (reader.next(key, value)) {
      println(value.toString + " belongs to cluster " + key.toString)
    }
    reader.close()
  }
  
  def writePointsToFile(points : ArrayList[Vector], 
      filename : String, fs : FileSystem,
      conf : Configuration) : Unit = {
    val path = new Path(filename)
    val writer = new SequenceFile.Writer(fs, conf, path, 
      classOf[LongWritable], classOf[VectorWritable])
    var recnum = 0
    val vec = new VectorWritable()
    for (point <- points.toArray()) {
      vec.set(point.asInstanceOf[Vector])
      writer.append(new LongWritable(recnum), vec)
      recnum += 1
    }
    writer.close()
  }
  
  def getPoints(daten : Array[(Double,Double)]) : ArrayList[Vector] = {
    var points = new ArrayList[Vector]()
    for (data <- daten) {
      val vec = new RandomAccessSparseVector(2)
      vec.assign(Array(data._1, data._2))
      points.add(vec)
    }
    points
  }
  
  @Test def testVectorize() = {
    val apples = ArrayBuffer[NamedVector]()
    apples += new NamedVector(new DenseVector(
      Array[Double](0.11, 510, 1)), "Small round green apple")
    apples += new NamedVector(new DenseVector(
      Array[Double](0.23, 650, 3)), "Large oval red apple")
    apples += new NamedVector(new DenseVector(
      Array[Double](0.09, 630, 1)), "Small elongated red apple")
    apples += new NamedVector(new DenseVector(
      Array[Double](0.25, 590, 2)), "Large round yellow apple")
    apples += new NamedVector(new DenseVector(
      Array[Double](0.18, 520, 2)), "Medium oval green apple")
    // vectorize (serialize vector)
    val conf = new Configuration()
    val fs = FileSystem.get(conf)
    val output = new Path("data/clustering/appledata/apples")
    val writer = new SequenceFile.Writer(fs, conf, output, 
      classOf[Text], classOf[VectorWritable])
    val vec = new VectorWritable()
    for (apple <- apples) {
      vec.set(apple)
      writer.append(new Text(apple.getName()), vec)
    }
    writer.close()
    // unvectorize (deserialize vectors)
    val reader = new SequenceFile.Reader(fs, 
      new Path("data/clustering/appledata/apples"), conf)
    val key = new Text()
    val value = new VectorWritable()
    while (reader.next(key, value)) {
      println(key.toString() + " => " + value.get().asFormatString())
    }
    reader.close()
  }
}