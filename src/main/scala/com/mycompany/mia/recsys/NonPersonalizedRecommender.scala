package com.mycompany.mia.recsys

import java.io.File

import scala.collection.JavaConversions.asScalaIterator

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel

/**
 * Given a movie ID, find top 5 movies like it using the 
 * following association metrics:
 *   basic = p(x AND y) / p(x)
 *   advanced = (p(x AND y) / p(x)) / (p(NOT x AND Y) / p(NOT x))
 */
class NonPersonalizedRecommender(ratingsFile: File) {

  val model = new FileDataModel(ratingsFile)

  /**
   * This is the method that will be called from the client.
   * @param movieID a movieID for which similar movies need
   *                to be found.
   * @param topN    the number of most similar movies to be
   *                returned (minus the movieID itself).
   * @param simfunc a function that returns the similarity 
   *                between two movieIDs.
   * @return a List of (movieID, similarity) tuples for the
   *                topN similar movies to movieID.
   */
  def similarMovies(movieId: Long, topN: Int, 
      simfunc: (Long, Long) => Double): List[(Long,Double)] = {
    val items = model.getItemIDs()
    val similarItems = items.filter(item => item != movieId)
      .map(item => (item.asInstanceOf[Long], 
         simfunc.apply(movieId, item)))
      .toList
      .sortWith((a, b) => a._2 > b._2)
      .slice(0, topN)
    similarItems
  }
  
  /**
   * Models basic association similarity. This function is
   * passed in from the similarMovies() method by client.
   * @param x a movieID.
   * @param y another movieID.
   * @return the similarity score between x and y.
   */
  def basicSimilarity(x: Long, y: Long): Double = {
    val xprefs = binaryRatings(x)
    val yprefs = binaryRatings(y)
    val pXY = numAssociations(xprefs, yprefs).toDouble
    val pX = xprefs.filter(x => x).size.toDouble
    pXY / pX
  }
  
  /**
   * Models advanced association similarity. This function is
   * passed in from the similarMovies() method by client.
   * @param x a movieID.
   * @param y another movieID.
   * @return the similarity score between x and y.
   */
  def advancedSimilarity(x: Long, y: Long): Double = {
    val xprefs = binaryRatings(x)
    val yprefs = binaryRatings(y)
    val notXprefs = xprefs.map(pref => !pref)
    val pXY = numAssociations(xprefs, yprefs).toDouble
    val pX = xprefs.filter(x => x).size.toDouble
    val pNotXY = numAssociations(notXprefs, yprefs).toDouble
    val pNotX = notXprefs.filter(x => x).size.toDouble
    (pXY / pX) / (pNotXY / pNotX)
  }

  /**
   * Given a item (movie ID), converts the preference array of
   * (userID:rating) elements to a Boolean List of true if user
   * has a rating and false if not.
   * @param item the movieID.
   * @return List[Boolean] of "binary" true/false ratings.
   */
  def binaryRatings(item: Long): List[Boolean] = {
    model.getUserIDs()
      .map(user => model.getPreferenceValue(user, item) != null)
      .toList
  }
  
  /**
   * Calculates the number of associations between Boolean lists
   * xs and ys. This differs from intersection in that we only 
   * count if corresponding elements of xs and ys are both true.
   * @param xs a list of binary ratings for all users.
   * @param ys another list of binary ratings for all users.
   * @return a count of where both elements of xs and ys are true.
   */
  def numAssociations(xs: List[Boolean], ys: List[Boolean]): Int = {
    xs.zip(ys).filter(xy => xy._1 && xy._2).size
  }
}