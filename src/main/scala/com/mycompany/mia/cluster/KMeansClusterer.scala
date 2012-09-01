package com.mycompany.mia.cluster

import java.util.HashSet

import scala.collection.JavaConversions.{bufferAsJavaList, asScalaBuffer}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{Text, IntWritable, SequenceFile}
import org.apache.mahout.clustering.canopy.CanopyClusterer
import org.apache.mahout.clustering.dirichlet.UncommonDistributions
import org.apache.mahout.clustering.iterator.ClusterWritable
import org.apache.mahout.clustering.kmeans.{Kluster, KMeansDriver}
import org.apache.mahout.common.distance.EuclideanDistanceMeasure
import org.apache.mahout.math.{VectorWritable, Vector, DenseVector}

object KMeansClusterer extends App {

  val INPUT_PATH = "data/clustering/input"
  val CLUSTER_PATH = "data/clustering/clusters"
  val OUTPUT_PATH = "data/clustering/output"

  val conf = new Configuration()
  val fs = FileSystem.get(conf)

  // generate sample data
  val sampleData = new ArrayBuffer[Vector]()
  sampleData.addAll(generateSamples(400, 1, 1, 3))
  sampleData.addAll(generateSamples(300, 1, 0, 0.5))
  sampleData.addAll(generateSamples(300, 0, 2, 0.1))
  writePointsToFile(sampleData, INPUT_PATH, fs, conf)
  
  // find initial centroids
  val centroids = args(0) match {
    case "random" => getInitialRandomCentroids(sampleData, 3)
    case "canopy" => getInitialCanopyCentroids(sampleData, 3.0, 1.5)
    case _ => throw new IllegalArgumentException(
      "centroid finder can only be random or canopy")
  }
  writeClustersToFile(centroids, CLUSTER_PATH + "/part-00000", 
    fs, conf)
  
  KMeansDriver.run(
      new Path(INPUT_PATH),
      new Path(CLUSTER_PATH),
      new Path(OUTPUT_PATH),
      new EuclideanDistanceMeasure(),
      0.01, 10, true, 0.5, false)
  
  // read off the clusters (output is in clusters-10-final
  // since we asked for 10 iterations
  // NOTE: sometimes KMeans will converge before 10, in that
  // case this part will crash. This should preferably be refactored
  // to its own little application. But it seems that Mahout expects
  // people to use scripts mostly, so this is probably not that 
  // important to do... 
  val reader = new SequenceFile.Reader(fs, 
    new Path(OUTPUT_PATH + "/clusters-10-final/part-r-00000"),
    conf)
  val key = new IntWritable()
  val value = new ClusterWritable()
  while (reader.next(key, value)) {
    println("Cluster " + key.toString + " contains " + 
      value.getValue().asFormatString(null))
  }
  reader.close()
  
  //////////////// methods //////////////////
  
  def getInitialRandomCentroids(sampleData : ArrayBuffer[Vector], 
      k : Integer) : ArrayBuffer[Kluster] = {
    var clusterId = 0
    val clusters = new ArrayBuffer[Kluster]()
    val centroids = getRandomPoints(sampleData, k)
    for (centroid <- centroids) {
      clusters += new Kluster(centroid, clusterId, 
        new EuclideanDistanceMeasure())
      clusterId += 1
    }
    clusters
  }
  
  def getInitialCanopyCentroids(
      sampleData : ArrayBuffer[Vector],
      t1 : Double, t2 : Double) : ArrayBuffer[Kluster] = {
    val sampleDataList = new java.util.ArrayList[Vector]()
    sampleDataList.addAll(sampleData)
    val canopies = CanopyClusterer.createCanopies(sampleDataList, 
      new EuclideanDistanceMeasure(), t1, t2)
    val clusters = new ArrayBuffer[Kluster]()
    var clusterId = 0
    for (canopy <- canopies) {
      clusters += new Kluster(canopy.getCenter(), clusterId, 
        new EuclideanDistanceMeasure())
      clusterId += 1
    }
    clusters
  }
  
  def generateSamples(num : Integer, mx : Double, 
      my : Double, sd : Double) : ArrayBuffer[Vector] = {
    val vectors = new ArrayBuffer[Vector]()
    for (i <- 0 to num) {
      vectors += new DenseVector(Array[Double](
        UncommonDistributions.rNorm(mx, sd),
        UncommonDistributions.rNorm(my, sd)
      ))
    }
    vectors
  }

  def getRandomPoints(sampleData : ArrayBuffer[Vector], k : Integer) : 
      ArrayBuffer[Vector] = {
    val n = sampleData.size
    val alreadySeen = new HashSet[Integer]()
    val rand = new Random(System.currentTimeMillis)
    val randoms = new ArrayBuffer[Integer]()
    do {
      val nextint = rand.nextInt(n)
      if (! alreadySeen.contains(nextint)) {
        randoms.append(nextint)
        alreadySeen.add(nextint)
      }
    } while (randoms.size() < 3)
    randoms.map(sampleData.get(_))
  }
  
  def writePointsToFile(vectors : ArrayBuffer[Vector],
      filename : String, fs : FileSystem, 
      conf : Configuration) : Unit = {
    val path = new Path(filename)
    val writer = new SequenceFile.Writer(fs, conf, path, 
      classOf[IntWritable], classOf[VectorWritable])
    var recnum = 0
    val vec = new VectorWritable()
    for (vector <- vectors) {
      vec.set(vector.asInstanceOf[Vector])
      writer.append(new IntWritable(recnum), vec)
      recnum += 1
    }
    writer.close()
  }
  
  def writeClustersToFile(clusters : ArrayBuffer[Kluster],
      filename : String, fs : FileSystem, 
      conf : Configuration) : Unit = {
    val path = new Path(filename)
    val writer = new SequenceFile.Writer(fs, conf, path, 
      classOf[Text], classOf[Kluster])
    for (cluster <- clusters) {
      writer.append(new Text(cluster.getIdentifier()), cluster)
    }
    writer.close()
  }
}