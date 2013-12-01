package com.mycompany.mia.recsys

import java.io.File
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import scala.collection.JavaConversions._
import org.apache.mahout.math.SparseMatrix
import org.apache.mahout.math.RandomAccessSparseVector
import org.apache.mahout.math.function.DoubleDoubleFunction
import org.apache.mahout.math.Vector
import org.apache.mahout.math.Matrix

/**
 * Predict a rating that a user would give a movie by looking
 * at ratings given by this user to other movies like this
 * one.
 */
class ItemItemCollaborativeFilteringRecommender(modelfile: File) {

  val model = new FileDataModel(modelfile)
  val itemIndex = model.getItemIDs().zipWithIndex.toMap
  val userIndex = model.getUserIDs().zipWithIndex.toMap
  val iiModel = buildIIModel()
  val userMeans = iiModel._1
  val mcRatings = iiModel._2
  val itemSims = iiModel._3

  /**
   * Compute a predicted rating using the following formula:
   *                   sum(sim(I,J)*(r(J))
   * P(u,I) = mu(u) + ---------------------
   *                     sum(|sim(I,J)|
   * summing over all items J in item neighborhood.
   * @param user the userID.
   * @param item the itemID.
   * @return the predicted rating for (userID,itemID).
   */
  def predictRating(user: Long, item: Long): Double = {
    val neighborhood = getItemNeighborhood(user, item, 20)
    val nd = neighborhood.map(itemsim => itemsim._1)
      .map(nitem => (
        mcRatings.get(userIndex(user), itemIndex(nitem)) * 
        itemSims.get(itemIndex(item), itemIndex(nitem)),
        math.abs(itemSims.get(itemIndex(item), itemIndex(nitem)))))
    val num = nd.map(x => x._1).foldLeft(0.0D)(_ + _)
    val den = nd.map(x => x._2).foldLeft(0.0D)(_ + _)
    userMeans.get(userIndex(user)) + (num / den)
  }
  
  /**
   * Find the topN similar items to the specified item.
   * @param item the itemID.
   * @param topN the number of most similar items.
   * @return List of (itemID, score) tuples.
   */
  def findSimilar(item: Long, topN: Int): List[(Long,Double)] = {
    model.getItemIDs()
      .filter(itemID => itemID != item)
      .map(itemID => (itemID.toLong, 
        itemSims.get(itemIndex(item), itemIndex(itemID))))
      .toList
      .sortWith((a, b) => a._2 > b._2)
      .slice(0, topN)
  }
  
  /**
   * Find other items rated by this user (except this
   * item), sort items by item similarity and return 
   * the top numNeighbors items.
   * @param user the user ID.
   * @param item the item ID.
   * @param numNeighbors number of neighbors to find.
   * @return List of (itemID, similarity) tuples.
   */
  def getItemNeighborhood(user: Long, item: Long, 
      numNeighbors: Int): List[(Long,Double)] = {
    model.getItemIDsFromUser(user)
      .filter(itemID => itemID != item)
      .map(itemID => (itemID.toLong, 
        itemSims.get(itemIndex(item), itemIndex(itemID))))
      .toList
      .sortWith((a, b) => a._2 > b._2)
      .slice(0, numNeighbors)
  }
  
  /**
   * Builds ItemItem model by mean centering user ratings,
   * then considering only items with ratings > 0, compute
   * item item similarities using cosine similarity.
   * @return a triple containing the user means, a matrix
   *         containing the mean centered ratings, and
   *         another matrix containing Item-Item cosine
   *         similarities.
   */
  def buildIIModel(): (Vector, Matrix, Matrix) = {
    // build rating matrix
    val ratingMatrix = new SparseMatrix(
      model.getNumUsers(), model.getNumItems())
    for (user <- model.getUserIDs()) {
      for (item <- model.getItemIDsFromUser(user)) {
        ratingMatrix.setQuick(userIndex(user), itemIndex(item), 
          model.getPreferenceValue(user, item).toDouble)
      }
    }
    // find user mean for each user and subtract it
    // then normalize so we can calculate cosine
    // similarity by doing matrix multiplication
    val userMeans = new RandomAccessSparseVector(model.getNumUsers())
    val normRatingMatrix = new SparseMatrix(
      model.getNumUsers(), model.getNumItems())
    for (user <- model.getUserIDs()) {
      val userRow = ratingMatrix.viewRow(userIndex(user))
      val len = userRow.all().filter(e => e != null).size.toDouble
      val sum = userRow.zSum()
      val userMean = sum / len
      userMeans.setQuick(userIndex(user), userMean)
      normRatingMatrix.assignRow(userIndex(user), 
        userRow.assign(new PlusIfPositive, userMean))
    }
    for (item <- model.getItemIDs()) {
      val itemCol = normRatingMatrix.viewColumn(itemIndex(item))
      val norm = itemCol.norm(2.0D)
      normRatingMatrix.assignColumn(itemIndex(item), 
        itemCol.times(1.0D / norm))
    }
    val itemsims = normRatingMatrix.transpose().times(normRatingMatrix)
    (userMeans, normRatingMatrix, itemsims)    
  }
  
  class PlusIfPositive extends DoubleDoubleFunction {
    override def apply(elem: Double, other: Double): Double = {
      if (elem > 0.0D) elem - other
      else 0.0D
    }
  }
}