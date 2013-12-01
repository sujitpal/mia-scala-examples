package com.mycompany.mia.recsys

import java.io.File
import org.junit.Test
import org.junit.Assert

class ItemItemCollaborativeFilteringRecommenderTest {

  val modelfile = new File("data/recsys/ratings.csv")
  val testUserItems = List((1024, 77), (1024, 268), (1024, 462), 
      (1024, 393), (1024, 36955), (2048, 77), 
      (2048, 36955), (2048, 788))
  val expectedRatings = List(4.1968, 2.3366, 2.9900,
    3.7702, 2.5612, 4.5102, 3.8545, 4.1253)
  val expectedSimilarityResults = List((550, 0.3192), (629, 0.3078),
      (38, 0.2574), (278, 0.2399), (680, 0.2394))
  
  val iicf = new ItemItemCollaborativeFilteringRecommender(modelfile)
  
  @Test def testPredictRating(): Unit = {
    var i = 0
    testUserItems.foreach(userItem => {
      val rating = iicf.predictRating(userItem._1, userItem._2)
      val expectedRating = expectedRatings(i)
      Console.println("Pred(%d:%d) = actual %f, expected %f".format(
        userItem._1, userItem._2, rating, expectedRating))
//      Assert.assertEquals(rating, expectedRating, 0.01)
      i = i + 1
    })
  }
  
  @Test def testFindSimilar(): Unit = {
    val similarItems = iicf.findSimilar(77L, 5)
    var i = 0
    similarItems.foreach(similarItem => {
      val actualItem = similarItem._1
      val actualSimilarity = similarItem._2
      val expectedItem = expectedSimilarityResults(i)._1
      val expectedSimilarity = expectedSimilarityResults(i)._2
      Console.println("Found (%d,%f), expected (%d,%f)".format(
        actualItem, actualSimilarity, expectedItem, expectedSimilarity))
//      Assert.assertEquals(actualItem, expectedItem)
//      Assert.assertEquals(actualSimilarity, expectedSimilarity, 0.01D)
      i = i + 1
    })
  }
}