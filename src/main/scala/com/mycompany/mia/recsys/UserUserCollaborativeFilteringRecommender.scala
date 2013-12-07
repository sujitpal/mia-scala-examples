package com.mycompany.mia.recsys

import java.io.File

import scala.collection.JavaConversions.asScalaIterator
import scala.collection.JavaConversions.iterableAsScalaIterable

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.mahout.math.Vector

/**
 * Predict a rating that a user would give a movie by looking
 * at predictions made by users most similar to the user.
 */
class UserUserCollaborativeFilteringRecommender(modelfile: File) {

  val model = new FileDataModel(modelfile)
  val itemIndex = model.getItemIDs().zipWithIndex.toMap
  val userIndex = model.getUserIDs().zipWithIndex.toMap
  
  /**
   * Compute a predicted rating using the following
   * formula:
   *                    sum(sim(u,v) * (r(v,i) - mu(v))
   *   p(u,i) = mu(u) + --------------------------------
   *                           sum(|sim(u,v)|)
   * sum is over all users v in neighborhood.
   * @param user the user ID
   * @param item the item ID
   * @return a predicted rating for (userID,itemID).
   */
  def predictRating(user: Long, item: Long): Double = {
    val muU = meanRating(user)
    val vectorU = centerUserVector(user, muU) 
    val neighbors = getUserNeighborhood(user, item, vectorU, 30)
    val ndpairs = neighbors.map(usersim => {
      val otheruser = usersim._1
      val simUV = usersim._2
      val muV = meanRating(otheruser)
      val rVI = model.getPreferenceValue(otheruser, item)
      (simUV * (rVI - muV), scala.math.abs(simUV))
    })
    val numer = ndpairs.map(x => x._1).foldLeft(0.0D)(_ + _)
    val denom = ndpairs.map(x => x._2).foldLeft(0.0D)(_ + _)
    muU + (numer / denom)
  }
  
  /**
   * Returns a neighborhood of similar users to a user
   * for a given item. Similarity metric used is Cosine
   * Similarity.
   * @param user the userID.
   * @param item the itemID.
   * @param vectorU the mean centered user vector for
   *        specified user.
   * @param nnbrs the number of neighbors to return.
   * @return a List of (userID,similarity) tuples for
   *        users in the neighborhood.
   */
  def getUserNeighborhood(user: Long, item: Long,
      vectorU: Vector,
      nnbrs: Int): List[(Long,Double)] = {
    model.getPreferencesForItem(item)
      // for the item, find all users that have rated the item
      // except the user itself.
      .map(pref => pref.getUserID().toLong)
      .filter(_ != user)
      // then mean center that rating and compute 
      // the cosine similarity between this user 
      // and our user
      .map(otheruser => {
        val muV = meanRating(otheruser)
        val vectorV = centerUserVector(otheruser, muV)
        (otheruser, cosineSimilarity(vectorU, vectorV))
      })
      .toList
      // sort by similarity and return the topN
      .sortWith((a,b) => a._2 > b._2)
      .slice(0, nnbrs)
  }
  
  /**
   * Calculate the mean rating for a user.
   * @param user the userID.
   * @return the mean user rating for that user.
   */
  def meanRating(user: Long): Double = {
    val ratings = model.getPreferencesFromUser(user)
      .map(pref => pref.getValue())
      .toList
    ratings.foldLeft(0.0D)(_ + _) / ratings.size
  }
  
  /**
   * Build a vector of item ratings for the user and
   * center them around the mean specified.
   * @param user the userID.
   * @param meanRating the mean item rating for user.
   * @return a vector containing mean centered ratings.
   */
  def centerUserVector(user: Long, meanRating: Double): Vector = {
    val uservec = new RandomAccessSparseVector(itemIndex.size)
    model.getPreferencesFromUser(user)
      .foreach(pref => uservec.setQuick(
        itemIndex(pref.getItemID()), 
        pref.getValue() - meanRating))
    uservec
  }

  /**
   * Compute cosine similarity between user vectors.
   * The isNAN() check is for cases where the user
   * has rated everything the same, so the mean
   * centered vector is all zeros, and the norm(2)
   * is also zero. The correct behavior (based on
   * np.linalg.norm()) is to return 0.0 in that case.
   * @param u vector for user u
   * @param v vector for user v
   * @return the cosine similarity between vectors.
   */
  def cosineSimilarity(u: Vector, v: Vector): Double = {
    val cosim = u.dot(v) / (u.norm(2) * v.norm(2))
    if (cosim.isNaN()) 0.0D else cosim
  }
}