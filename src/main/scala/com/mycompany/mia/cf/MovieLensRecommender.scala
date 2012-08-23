package com.mycompany.mia.cf

import java.io.File
import java.util.List

import scala.collection.JavaConversions.asScalaBuffer
import scala.collection.mutable.HashSet
import scala.io.Source

import org.apache.mahout.cf.taste.common.Refreshable
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.model.DataModel
import org.apache.mahout.cf.taste.recommender.{Recommender, RecommendedItem, IDRescorer}

object MovieLensRecommenderRunner extends App {
  // grab the input file name
  val filename = if (args.length == 1) args(0) else "unknown"
  if ("unknown".equals(filename)) {
    println("Please specify input file")
    System.exit(-1)
  }
  // train recommender
  val recommender = new MovieLensRecommender(
    new FileDataModel(new File(filename)))
  // test recommender
  val alreadySeen = new HashSet[Long]()
  val lines = Source.fromFile(filename).getLines
  for (line <- lines) {
    val user = line.split(",")(0).toLong
    if (! alreadySeen.contains(user)) {
      val items = recommender.recommend(user, 100)
      println(user + " =>" + items.map(x => x.getItemID).
        foldLeft("")(_ + " " + _))
    }
    alreadySeen += user
  }
}

class MovieLensRecommender(model : DataModel) extends Recommender {

  val similarity = new PearsonCorrelationSimilarity(model)
  val delegate = new GenericItemBasedRecommender(model, similarity)

  // everything below this is boilerplate. We could use the 
  // RecommenderWrapper if it was part of Mahout-Core but its part 
  // of the Mahout-Integration for the webapp
  
  def recommend(userID: Long, howMany: Int): List[RecommendedItem] = {
    delegate.recommend(userID, howMany)
  }

  def recommend(userID: Long, howMany: Int, rescorer: IDRescorer): List[RecommendedItem] = {
    delegate.recommend(userID, howMany, rescorer)
  }

  def estimatePreference(userID: Long, itemID: Long): Float = {
    delegate.estimatePreference(userID, itemID)
  }

  def setPreference(userID: Long, itemID: Long, value: Float): Unit = {
    delegate.setPreference(userID, itemID, value)
  }

  def removePreference(userID: Long, itemID: Long): Unit = {
    delegate.removePreference(userID, itemID)
  }

  def getDataModel(): DataModel = {
    delegate.getDataModel()
  }

  def refresh(alreadyRefreshed: java.util.Collection[Refreshable]): Unit = {
    delegate.refresh(alreadyRefreshed)
  }
}
