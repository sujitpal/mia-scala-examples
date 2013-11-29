package com.mycompany.mia.recsys

import java.io.File

import org.junit.Assert
import org.junit.Test

class NonPersonalizedRecommenderTest {

  val testMovieIds = List(11, 121, 8587)
  val simpleResults = Map[Long,List[(Long,Double)]](
    11L   -> List((603,0.96), (1892,0.94), (1891,0.94), 
               (120,0.93), (1894,0.93)),
    121L  -> List((120,0.95), (122,0.95), (603,0.94), 
               (597,0.89), (604,0.88)),
    8587L -> List((603,0.92), (597,0.90), (607,0.87),
               (120,0.86), (13,0.86)))
  val advancedResults = Map[Long,List[(Long,Double)]](
    11L   -> List((1891,5.69), (1892,5.65), (243,5.00),
               (1894,4.72), (2164,4.11)),
    121L  -> List((122,4.74), (120,3.82), (2164,3.40),
               (243,3.26), (1894,3.22)),
    8587L -> List((10020,4.18), (812,4.03), (7443,2.63),
               (9331,2.46), (786,2.39)))
  
  @Test def testBasicSimilarity(): Unit = {
    val npr = new NonPersonalizedRecommender(new File(
      "data/recsys/ratings.csv"))
    testMovieIds.foreach(movieId => {
      val similarMoviesWithBasicSimilarity = npr.similarMovies(
        movieId, 5, npr.basicSimilarity)
      Console.println(movieId + " => " + similarMoviesWithBasicSimilarity)
      Console.println("expected: " + simpleResults(movieId))
      assertEquals(similarMoviesWithBasicSimilarity, simpleResults(movieId))
    })
  }
  
  @Test def testAdvancedSimilarity(): Unit = {
    val npr = new NonPersonalizedRecommender(new File(
      "data/recsys/ratings.csv"))
    testMovieIds.foreach(movieId => {
      val similarMoviesWithAdvancedSimilarity = npr.similarMovies(
        movieId, 5, npr.advancedSimilarity)
      Console.println(movieId + " => " + similarMoviesWithAdvancedSimilarity)
      Console.println("expected: " + advancedResults(movieId))
      assertEquals(similarMoviesWithAdvancedSimilarity, advancedResults(movieId))
    })
  }
  
  def assertEquals(actual: List[(Long,Double)], 
      expected: List[(Long,Double)]): Unit = {
    Assert.assertEquals(actual.size, expected.size)
    actual.zip(expected).foreach(aze => {
      Assert.assertEquals(aze._1._1, aze._2._1)
      Assert.assertEquals(aze._1._2, aze._2._2, 0.01)
    })
  }
}