package com.mycompany.mia.recsys

import java.io.File

import scala.collection.JavaConversions.asScalaIterator
import scala.collection.JavaConversions.iterableAsScalaIterable

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.mahout.math.SparseMatrix
import org.apache.mahout.math.Vector

/**
 * Predict a rating that a user would give a movie by looking
 * at predictions made by users most similar to the user.
 */
class UserUserCollaborativeFilteringRecommender(modelfile: File) {

  val model = new FileDataModel(modelfile)
  val userIndex = model.getUserIDs().zipWithIndex.toMap
  val reverseUserIndex = userIndex.map {case (k, v) => (v, k)}
  val itemIndex = model.getItemIDs().zipWithIndex.toMap
  val uuModel = buildUUModel()
  val userMeans = uuModel._1
  val mcRatings = uuModel._2
  val userSims = uuModel._3

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
    // find the 30 most similar users to this user
    // from this set
    val neighbors = getUserNeighborhood(user, item, 30)
    // calculate the rating and return
    val nd = neighbors.map(usersim => {
      val user = usersim._1
      val simUV = usersim._2
      val normRating = mcRatings.getQuick(
        userIndex(user), itemIndex(item))
      (normRating * simUV, math.abs(simUV))
      })
    val num = nd.map(x => x._1).foldLeft(0.0D)(_ + _)
    val den = nd.map(x => x._2).foldLeft(0.0D)(_ + _)
    userMeans.get(userIndex(user)) + (num / den)
  }
  
  /**
   * Scans the similarity matrix row for the provided user,
   * sorts the elements by descending order of similarity
   * and reports the top numNeighbors values with the
   * associated userIDs.
   * @param user the user for which similar users are to
   *             be found.
   * @param item the item for which similar users are to
   *             be found.
   * @param numNeighbors the number of neighbors to find.
   */
  def getUserNeighborhood(user: Long,  item: Long, 
      numNeighbors: Int): List[(Long,Double)] = {
    // for the item, find users (except this user)
    // who have rated the item
	val otherUsers = model.getUserIDs()
	  .map(user => (user, model.getItemIDsFromUser(user)))
	  .filter(useritems => useritems._2.contains(item))
	  .map(useritems => useritems._1.toLong)
	  .toSet
    userSims.viewRow(userIndex(user))
      .all()
      .zipWithIndex
      .map(simindex => (reverseUserIndex(simindex._2).toLong, 
        simindex._2.toDouble))
      .filter(usersim => otherUsers.contains(usersim._1))
      .toList
      .sortWith((a, b) => a._2 > b._2)
      .slice(0, numNeighbors)
  }
  
  /**
   * Builds a UU Model and returns a vector containing user's
   * mean ratings and a matrix of user-user similarities. The
   * method is called by the class during initialization.
   * @return tuple of user mean ratings and user-user
   *         similarities.
   */
  def buildUUModel(): (Vector, Matrix, Matrix) = {
    // build rating matrix
    val ratingMatrix = new SparseMatrix(
      model.getNumUsers(), model.getNumItems())
    for (user <- model.getUserIDs()) {
      for (item <- model.getItemIDsFromUser(user)) {
        ratingMatrix.setQuick(userIndex(user), itemIndex(item), 
          model.getPreferenceValue(user, item).toDouble)
      }
    }
    // mean center ratings by user, ie find mean for each row
    // and subtract it from that row. We then normalize each
    // row by its 2-norm (so we can compute cosine similarity 
    // calculation as normRatingMatrix * normRatingMatrix.T)
    val userMeans = new RandomAccessSparseVector(userIndex.size)
    val mcRatingMatrix = new SparseMatrix(
      model.getNumUsers(), model.getNumItems())
    val userNorms = new RandomAccessSparseVector(userIndex.size)
    for (r <- 0 until userIndex.size) {
      val row = ratingMatrix.viewRow(r)
      val len = row.all().filter(_ != null).size.toDouble
      val sum = row.zSum()
      val userMean = sum / len
      val norm = row.norm(2.0D)
      userMeans.setQuick(r, userMean)
      mcRatingMatrix.assignRow(r, row.plus(-1.0D * userMean))
    }
    // calculate cosine similarity for all users
    val normRatingMatrix = new SparseMatrix(
      model.getNumUsers(), model.getNumItems())
    for (r <- 0 until userIndex.size) {
      val row = mcRatingMatrix.viewRow(r)
      val norm = row.norm(2.0D)
      normRatingMatrix.assignRow(r, row.times(1.0D / norm))
    }
    val similarities = normRatingMatrix.times(
      normRatingMatrix.transpose())
    (userMeans, mcRatingMatrix, similarities)
  }
}