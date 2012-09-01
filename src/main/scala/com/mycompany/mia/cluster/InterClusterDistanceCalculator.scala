package com.mycompany.mia.cluster

import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.io.{Writable, SequenceFile}
import org.apache.mahout.clustering.iterator.ClusterWritable
import org.apache.mahout.clustering.Cluster
import org.apache.mahout.common.distance.CosineDistanceMeasure

/**
 * Called from command line like this:
 * hduser@cyclone:mahout-distribution-0.7$ hadoop \
 *   jar /tmp/my-mahout-fatjar.jar \
 *   com.mycompany.mia.cluster.InterClusterDistanceCalculator \
 *   reuters-kmeans-clusters/clusters-3-final/part-r-00000
 */
object InterClusterDistanceCalculator extends App {

  val conf = new Configuration()
  val inputPath = new Path(args(0))
  val fs = FileSystem.get(inputPath.toUri(), conf)
  val clusters = ArrayBuffer[Cluster]()
  val reader = new SequenceFile.Reader(fs, inputPath, conf)
  var key = reader.getKeyClass().newInstance().asInstanceOf[Writable]
  var value = reader.getValueClass().newInstance().asInstanceOf[Writable]
  while (reader.next(key, value)) {
    clusters += value.asInstanceOf[ClusterWritable].getValue()
    value = reader.getValueClass().newInstance().asInstanceOf[Writable]
  }
  val distMeasure = new CosineDistanceMeasure()
  var max = 0.0D
  var min = Double.MaxValue
  var sum = 0.0D
  var count = 0
  for (i <- 0 to (clusters.length - 1); 
       j <- (i + 1) to (clusters.length - 1)) {
    val dist = distMeasure.distance(
      clusters(i).getCenter(), clusters(j).getCenter())
    min = Math.min(dist, min)
    max = Math.max(dist, max)
    sum += dist
    count += 1
  }
  println("Maximum Intercluster Distance: " + max)
  println("Minimum Intercluster Distance: " + min)
  println("Average Intercluster Distance (scaled): " + 
    ((sum / count) - min) / (max - min))
}