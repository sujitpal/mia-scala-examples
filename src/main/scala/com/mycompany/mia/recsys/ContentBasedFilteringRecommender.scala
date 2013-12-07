package com.mycompany.mia.recsys

import java.io.File

import scala.actors.threadpool.AtomicInteger
import scala.collection.JavaConversions.asScalaIterator
import scala.collection.JavaConversions.iterableAsScalaIterable
import scala.io.Source

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.model.DataModel
import org.apache.mahout.math.ConstantVector
import org.apache.mahout.math.SparseMatrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.function.DoubleDoubleFunction
import org.apache.mahout.math.function.VectorFunction

/**
 * Recommend items that are similar to the ones user has 
 * expressed a preference for.
 */
class ContentBasedFilteringRecommender(
    modelfile: File, tagfile: File) {

  val model = new FileDataModel(modelfile)
  val itemIndex = model.getItemIDs().zipWithIndex.toMap  
  
  val tfidfBuilder = new TfIdfBuilder(model, itemIndex, tagfile)
  val tfidf = tfidfBuilder.build()
  val tagIndex = tfidfBuilder.tagIndex
  
  /**
   * Find movies similar to the ones that user has already
   * expressed a preference for.
   * @param user the userID.
   * @param topN the top N similar movies.
   * @return a List of (item,score) tuples.
   */
  def similarMovies(user: Long, topN: Int,
      profileFunc: (Long) => Vector): List[(Long,Double)] = {
    val userProfile = profileFunc.apply(user)
    val alreadyRated = model.getItemIDsFromUser(user).toSet
    val similarMovies = model.getItemIDs()
      .filter(item => (! alreadyRated.contains(item)))
      .map(item => (item, tfidf.viewRow(itemIndex(item))))
      .map(itemvector => (
        itemvector._1.toLong, 
        cosineSimilarity(userProfile, itemvector._2)))
      .toList
      .sortWith((a, b) => a._2 > b._2)
      .slice(0, topN)
    similarMovies
  }
  
  /**
   * Compute the user's tag profile. This is done by adding up
   * all the TFIDF item vectors for which the user rated >= 3.5.
   * @param user the userID for the user.
   * @return vector representing the user's tag profile.
   */
  def makeUnweightedUserProfile(user: Long): Vector = {
    val highlyRatedItemVectors = model.getItemIDsFromUser(user)
      .map(item => (item, model.getPreferenceValue(user, item)))
      .filter(itemrating => itemrating._2 >= 3.5D)
      .map(itemrating => itemrating._1)
      .map(item => tfidf.viewRow(itemIndex(item)))
      .toList
    val numTags = tagIndex.size
    val zeros = new ConstantVector(0.0D, numTags).asInstanceOf[Vector]
    highlyRatedItemVectors.foldLeft(zeros)((a, b) => a.plus(b))
  }
  
  /**
   * Compute the average rating for the user, then calculate
   * weights for each item rating based on the sum of the
   * deviance of the rating from the user mean. Compute the
   * sum of item tags as before but this time weight each
   * item vector with this computed weight.
   * @param user the userID.
   * @return the weighted user profile.
   */
  def makeWeightedUserProfile(user: Long): Vector = {
    val ratings = model.getItemIDsFromUser(user)
      .map(item => model.getPreferenceValue(user, item))
      .toList
    val mu = ratings.foldLeft(0.0D)(_ + _) / ratings.size.toDouble
    val weights = model.getItemIDsFromUser(user)
      .map(item => model.getPreferenceValue(user, item) - mu)
      .toList
    val ratingVectors = model.getItemIDsFromUser(user)
      .map(item => tfidf.viewRow(itemIndex(item)))
      .toList
    val weightedRatingVector = weights.zip(ratingVectors)
      .map(wv => wv._2.times(wv._1))
    val numTags = tagIndex.size
    val zeros = new ConstantVector(0.0D, numTags).asInstanceOf[Vector]
    weightedRatingVector.foldLeft(zeros)((a, b) => a.plus(b))
  }
  
  /**
   * Compute cosine similarity between the user vector and
   * each item vector.
   * @param u the user vector.
   * @param v the item vector.
   * @return the cosine similarity between u and v.
   */
  def cosineSimilarity(u: Vector, v: Vector): Double = {
    u.dot(v) / (u.norm(2) * v.norm(2))
  }
}

/**
 * Model to convert item tags into a Tag-Movie (TD) matrix,
 * which is then normalized to TFIDF and normalized across
 * items (movies).
 * @param model the DataModel.
 * @param itemIndex the itemID to row index mapping.
 * @param tagfile the File containing the (itemID, tag) tuples.
 */
class TfIdfBuilder(val model: DataModel, 
    val itemIndex: Map[java.lang.Long,Int], val tagfile: File) {
  
  val tagIndex = collection.mutable.Map[String,Int]()

  /**
   * Returns a Sparse Matrix of movies vs tags, with the tag
   * frequency TFIDF normalized and each item vector further
   * unit normalized. Also builds the tagIndex as a side effect.
   * @return SparseVector of movies vs tags.
   */
  def build(): SparseMatrix = { 
    // assign each tag an id
    val tagIds = new AtomicInteger(0)
    Source.fromFile(tagfile)
      .getLines()
      .foreach(line => {
        val tag = line.split(",")(1)
        if (! tagIndex.contains(tag)) 
          tagIndex(tag) = tagIds.getAndIncrement()
      })
    // populate the SparseMatrix in the second scan through tagfile
    val movieTagMatrix = new SparseMatrix(itemIndex.size, tagIndex.size)
    Source.fromFile(tagfile)
      .getLines()
      .foreach(line => {
        val columns = line.split(",")
        val row = itemIndex(columns(0).toLong)
        val col = tagIndex(columns(1))
        movieTagMatrix.setQuick(row, col, 
          movieTagMatrix.getQuick(row, col) + 1.0D)
      })
    // we got our TF (raw term freqs), now find IDFs
    val numdocs = movieTagMatrix.numRows()
    val docfreq = movieTagMatrix.aggregateColumns(
      new AddOneIfNonZeroFunc)
    val idf = docfreq.assign(new IdfFunc, numdocs)    
    // now calculate TF-IDF
    for (r <- 0 until movieTagMatrix.numRows()) {
      val row = movieTagMatrix.viewRow(r).times(idf)
      movieTagMatrix.assignRow(r, row)
    }
    // then unit-normalize over each item (use Eucledian norm)
    val enorm = movieTagMatrix.aggregateRows(new SumOfSquaresFunc())
    for (r <- 0 until movieTagMatrix.numRows()) {
      val row = movieTagMatrix.viewRow(r).times(1.0D / enorm.get(r))
      movieTagMatrix.assignRow(r, row)
    }
    movieTagMatrix
  }

  /**
   * Sums the elements in a vector.
   */
  class SumOfSquaresFunc extends VectorFunction {
    override def apply(vec: Vector): Double = vec.norm(2)  
  }
  
  /**
   * Calculate number of documents in which a tag appears.
   * Document needs to be computed once regardless of the
   * number of tags.
   */
  class AddOneIfNonZeroFunc extends VectorFunction {
    override def apply(vec: Vector): Double = 
      vec.all().filter(e => e.get() > 0.0D).size.toDouble
  }
  
  /**
   * Given the number of documents per tag and the number of
   * documents (movies), calculates the IDF as 
   * log(numDocPerTag)/numDocs.
   */
  class IdfFunc extends DoubleDoubleFunction {
    override def apply(elem: Double, ext: Double): Double = {
        scala.math.log(elem / ext)
    }
  }
}