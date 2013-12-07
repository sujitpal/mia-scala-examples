package com.mycompany.mia.recsys

import java.io.File
import org.junit.Test
import org.junit.Assert

class UserUserCollaborativeFilteringRecommenderTest {

  val modelfile = new File("data/recsys/ratings.csv")
  
  val ratingTriples = List(
    (1024, 77,    4.3848),
    (1024, 268,   2.8646),
    (1024, 393,   3.8722),
    (1024, 462,   3.1082),
    (1024, 36955, 2.3524),
    (2048, 77,    4.8493),
    (2048, 788,   3.8509),
    (2048, 36955, 3.9698))
  
  @Test def testPredictions(): Unit = {
    val uucf = new UserUserCollaborativeFilteringRecommender(
      modelfile)
    ratingTriples.foreach(rt => { 
      val score = uucf.predictRating(rt._1, rt._2)
      Console.println("Pred(%d:%d) = actual %f, expected %f"
        .format(rt._1, rt._2, score, rt._3))
      Assert.assertEquals(score, rt._3, 0.01D)
    })
  }
}