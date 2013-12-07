package com.mycompany.mia.recsys

import java.io.File
import org.junit.Test
import org.junit.Assert

class ItemItemCollaborativeFilteringRecommenderTest {

  val modelfile = new File("data/recsys/ratings.csv")
  
  val predictRatingData = List(
    (1024, 77,    4.1968),
    (1024, 268,   2.3366),
    (1024, 393,   3.7702),
    (1024, 462,   2.9900),
    (1024, 36955, 2.5612),
    (2048, 77,    4.5102),
    (2048, 788,   4.1253),
    (2048, 36955, 3.8545))
  val findSimilarData = List(
    (550, 0.3192),
    (629, 0.3078),
    (38,  0.2574),
    (278, 0.2399),
    (680, 0.2394))
  
  val iicf = new ItemItemCollaborativeFilteringRecommender(
    modelfile)
  
  @Test def testPredictRating(): Unit = {
    predictRatingData.foreach(rating => {
      val predicted = iicf.predictRating(rating._1, rating._2)
      Console.println("Pred(%d:%d) = actual %f, expected %f".format(
        rating._1, rating._2, predicted, rating._3))
      Assert.assertEquals(predicted, rating._3, 0.01)
    })
  }
  
  @Test def testFindSimilar(): Unit = {
    val similarItems = iicf.findSimilar(77L, 5)
    var i = 0
    similarItems.foreach(similarItem => {
      val actualItem = similarItem._1
      val actualSimilarity = similarItem._2
      val expectedItem = findSimilarData(i)._1
      val expectedSimilarity = findSimilarData(i)._2
      Console.println("Found (%d,%f), expected (%d,%f)".format(
        actualItem, actualSimilarity, expectedItem, expectedSimilarity))
      Assert.assertEquals(actualItem, expectedItem)
      Assert.assertEquals(actualSimilarity, expectedSimilarity, 0.01D)
      i = i + 1
    })
  }
}