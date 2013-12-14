package com.mycompany.mia.recsys

import java.io.File

import scala.io.Source

import org.junit.Assert
import org.junit.Test

class LuceneIICFRecommenderTest {

  val liicf = new LuceneIICFRecommender(
    new File("data/recsys/ratings.csv"), 
    new File("data/recsys/movie-tags.csv"),
    new File("data/recsys/itemindex"))
  val titles = new File("data/recsys/movie-titles.csv")
  val movieNames = Source.fromFile(titles)
    .getLines()
    .map(line => {
      val Array(movieID, title) = line.split(",")
      (movieID.toLong, title)
    }).toMap
  
  @Test def testRecommendItemsGivenUser(): Unit = {
    val scores = liicf.recommend(15L, 20)
    Assert.assertEquals(scores.size, 20)
    Console.println("Recommendations for user(15)")
    scores.foreach(score => {
      Console.println("%5.3f %5d %s".format(
        score._2, score._1, movieNames(score._1)))
    })
  }
  
  @Test def testRecommendItemsGivenItem(): Unit = {
    val recommendedItems = liicf.similarItems(77L, 10)
    Assert.assertEquals(recommendedItems.size, 10)
    Console.println("recommendations for [%5d %s]"
      .format(77L, movieNames(77L)))
    recommendedItems.foreach(docsim => {
      Console.println("%7.4f %5d %s"
        .format(docsim._2, docsim._1, movieNames(docsim._1)))
    })
  }

  @Test def testPredictRatingForItem(): Unit = {
    val predictedRating = liicf.predict(2048L, 393L)
    Console.println("prediction(2048,393) = " + predictedRating)
    Assert.assertTrue(predictedRating > 4.0D)
    val predictRatingForRatedItem = liicf.predict(2048L, 77L)
    Console.println("prediction(2048,77) = " + predictedRating)
    Assert.assertEquals(predictedRating, 4.5D, 0.1D)
  }
}