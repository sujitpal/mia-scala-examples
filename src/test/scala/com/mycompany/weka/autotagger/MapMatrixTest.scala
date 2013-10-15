package com.mycompany.weka.autotagger;

import org.junit.{Assert, Test}

class MapMatrixTest {
    
  @Test def testBuildMapVectorFromString(): Unit = {
    val vector = MapVector.fromString(
      "5351617$2.35 8002744$1.10 5354318$2.12")
    Assert.assertEquals(3, vector.realSize(), 0)
    Assert.assertEquals(2.35, vector.elementValue("5351617"), 0.01)
  }

  @Test def testBuildNormalizedMapVectorFromString(): Unit = {
    val vector = MapVector.fromString(
      "5351617$2.35 8002744$1.10 5354318$2.12", true)
    Assert.assertEquals(3, vector.realSize(), 0)
    Assert.assertEquals(0.701, vector.elementValue("5351617"), 0.01)
  }

  @Test def testBuildMapVectorFromMap(): Unit = {
    val mapvector = new MapVector(
      Map(("1193", 5), ("661", 3), ("914", 3)))
    Assert.assertEquals(0, mapvector.elementValue("41"), 0)
    Assert.assertEquals(5, mapvector.elementValue("1193"), 0)
  }
  
  @Test def testMapVectorRealSize(): Unit = {
    val mapvector = new MapVector(
      Map(("1193", 5), ("661", 3), ("914", 3)))
    Assert.assertEquals(3, mapvector.realSize)
  }
  
  @Test def testComputeL2Norm(): Unit = {
    val mapvector = new MapVector(
      Map(("1193", 5), ("661", 3), ("914", 3)))
    Assert.assertEquals(6.557, mapvector.l2norm, 0.01)
  }

  @Test def testComputeDotProduct(): Unit = {
    val mapvector = new MapVector(
      Map(("1193", 5), ("661", 3), ("914", 3)))
    val mapvector2 = new MapVector(
      Map(("1357", 4), ("647", 4), ("661", 4)))
    val dotProduct = mapvector.dotProduct(mapvector2)
    Assert.assertEquals(12, dotProduct, 0.01)
  }
  
  @Test def testComputeCosineSimilarity(): Unit = {
    val mapvector = new MapVector(
      Map(("1193", 5), ("661", 3), ("914", 3)))
    val mapvector2 = new MapVector(
      Map(("1357", 4), ("647", 4), ("661", 4)))
    val cosim = mapvector.cosim(mapvector2)
    Assert.assertEquals(0.264, cosim, 0.01)
  }
  
  @Test def testComputeCentroid(): Unit = {
    val mapmatrix = new MapMatrix()
    mapmatrix.addVector(
      new MapVector(Map(("1193", 5), ("661", 3), ("914", 3))))
    mapmatrix.addVector(
      new MapVector(Map(("1357", 4), ("647", 4), ("661", 4))))
    mapmatrix.addVector(
      new MapVector(Map(("3421", 2), ("164", 4), ("914", 2))))
    val centroid = mapmatrix.centroid()
    Assert.assertEquals(7, centroid.realSize(), 0)
    Assert.assertEquals(1.667, centroid.elementValue("1193"), 0.01)
  }
}
