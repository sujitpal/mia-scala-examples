package com.mycompany.mia.recsys

import java.io.File

import org.junit.Assert
import org.junit.Test

class ContentBasedFilteringRecommenderTest {

  val modelfile = new File("data/recsys/ratings.csv")
  val tagfile = new File("data/recsys/movie-tags.csv")
  
  val testUsers = List(4045, 144, 3855, 1637, 2919)
  val expectedUnweightedResults = Map(
    4045L -> List((11L, 0.3596), (63L, 0.2612), (807L, 0.2363),
                  (187L, 0.2059), (2164L, 0.1899)),
    144L  -> List((11L, 0.3715), (585L, 0.2512), (38L, 0.1908),
                  (141L, 0.1861), (807L, 0.1748)),
    3855L -> List((1892L, 0.4303), (1894L, 0.2958), (63L, 0.2226),
                  (2164L, 0.2119), (604L, 0.1941)),
    1637L -> List((2164L, 0.2272), (141L, 0.2225), (745L, 0.2067),
                  (601L, 0.1995), (807L, 0.1846)),
    2919L -> List((11L, 0.3659), (1891L, 0.3278), (640L, 0.1958),
                  (424L, 0.1840), (180L, 0.1527)))
  val expectedWeightedResults = Map(
    4045L -> List((807L, 0.1932), (63L, 0.1438), (187L, 0.0947),
                  (11L, 0.0900), (641L, 0.0471)),
    144L  -> List((11L, 0.1394), (585L, 0.1229), (671L, 0.1130),
                  (672L, 0.0878), (141L, 0.0436)),
    3855L -> List((1892L, 0.2243), (1894L, 0.1465), (604L, 0.1258),
                  (462L, 0.1050), (10020L, 0.0898)),
    1637L -> List((393L, 0.1976), (24L, 0.1900), (2164L, 0.1522),
                  (601L, 0.1334), (5503L, 0.0992)),
    2919L  -> List((180L, 0.1454), (11L, 0.1238), (1891L, 0.1172),
                  (424L, 0.1074), (2501L, 0.0973)))      
  
  @Test def testUnweightedRecommendations(): Unit = {
    val cbf = new ContentBasedFilteringRecommender(
      new File("data/recsys/ratings.csv"),
      new File("data/recsys/movie-tags.csv"))
    testUsers.foreach(user => {
      val actualResults = cbf.similarMovies(user, 5, 
        cbf.makeUnweightedUserProfile)
      Console.println("results for user: " + user)
      Console.println("actuals=" + actualResults)
      Console.println("expected=" + expectedUnweightedResults(user))
//      assertEquals(actualResults, expectedUnweightedResults(user))
    })
  }

  @Test def testWeightedRecommendations(): Unit = {
    val cbf = new ContentBasedFilteringRecommender(
      new File("data/recsys/ratings.csv"),
      new File("data/recsys/movie-tags.csv"))
    testUsers.foreach(user => {
      val actualResults = cbf.similarMovies(user, 5, 
        cbf.makeWeightedUserProfile)
      Console.println("results for user: " + user)
      Console.println("actuals=" + actualResults)
      Console.println("expected=" + expectedWeightedResults(user))
//      assertEquals(actualResults, expectedWeightedResults(user))
    })
  }

  def assertEquals(xs: List[(Long,Double)], 
      ys: List[(Long,Double)]): Unit = {
    Assert.assertEquals(xs.size, ys.size)
    xs.zip(ys).foreach(xy => {
      Assert.assertEquals(xy._1._1, xy._2._1)
      Assert.assertEquals(xy._1._2, xy._2._2, 0.01D)
    })
  }
}