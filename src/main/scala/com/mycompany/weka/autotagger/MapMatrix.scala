package com.mycompany.weka.autotagger

import scala.Array.fallbackCanBuildFrom
import scala.actors.threadpool.AtomicInteger

class MapMatrix {

  var initVector = new MapVector(Map())
  var size = 0
  
  def addVector(vector: MapVector): Unit = {
    initVector = initVector.add(vector)
    size = size + 1
  }
  
  def centroid(): MapVector = {
    initVector.divide(size)
  }
}

object MapVector {
  
  def fromString(s: String): MapVector = {
    fromString(s, false)
  }
  
  def fromString(s: String, normalize: Boolean): MapVector = {
    val pairs = s.split(" ")
    val m: Map[String,Double] = pairs.map(pair => {
      val cols = pair.split("\\$")
      (cols(0), cols(1).toDouble)
    }).
    toMap
    val v = new MapVector(m) 
    if (normalize) v.divide(v.l2norm()) 
    else v
  }
  
  def toFormattedString(vector: MapVector): String = 
    vector.iterator.toList.
      map(pair => "%s$%-8.5f".format(pair._1, pair._2).trim()).
      mkString(" ")
}

class MapVector(val map: Map[String,Double]) {
  
  def iterator() = map.iterator
  
  def elementValue(key: String): Double = map.getOrElse(key, 0.0D)

  def divide(scalar: Double): MapVector =
    new MapVector(Map() ++
      iterator.toList.map(pair => (pair._1, pair._2 / scalar)))  
  
  def add(vector: MapVector): MapVector = {
    val keys = iterator.toList.map(pair => pair._1).
      union(vector.iterator.toList.map(pair => pair._1)).
      toSet
    new MapVector(Map() ++ keys.map(key => 
      (key, elementValue(key) + vector.elementValue(key))))
  }
  
  def dotProduct(vector: MapVector): Double = {
    val keys = iterator.toList.map(pair => pair._1).
      union(vector.iterator.toList.map(pair => pair._1)).
      toSet
    keys.map(key => elementValue(key) * vector.elementValue(key)).
      foldLeft(0.0D)(_ + _)
  }
  
  def l2norm(): Double = scala.math.sqrt(
    iterator.toList.
    map(pair => math.pow(pair._2, 2.0D)).
    foldLeft(0.0D)(_ + _))
  
  def cosim(vector: MapVector): Double =
    dotProduct(vector) / (l2norm() * vector.l2norm())  
  
  def realSize(): Int = map.size
  
  override def toString(): String = this.iterator.toList.toString
  
}