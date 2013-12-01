package com.mycompany.mia.recsys

import java.io.File

import org.junit.Test

class UserUserCollaborativeFilteringRecommenderTest {

  val modelfile = new File("data/recsys/ratings.csv")
  
  val testUserItemPairs = List((1024, 77), (1024, 268), 
      (1024, 462), (1024, 393), (1024, 36955), 
      (2048, 77), (2048, 36955), (2048, 788))
  val expectedRatings = List(3.8509, 3.9698, 4.8493, 
      3.1082, 3.8722, 2.3524, 4.3848, 2.8646)

  @Test def testPredictions(): Unit = {
    val uucf = new UserUserCollaborativeFilteringRecommender(
      modelfile)
    var i = 0
    testUserItemPairs.foreach(userItem => { 
      val score = uucf.predictRating(userItem._1, userItem._2)
      val expected = expectedRatings(i)
      i = i + 1
      Console.println("Pred(%d:%d) = actual %f, expected %f"
        .format(userItem._1, userItem._2, score, expected))
//      Assert.assertEquals(score, expectedRatings(i), 0.01D)
    })
  }
}