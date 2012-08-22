package com.mycompany.mia.cf

import java.io.File

import scala.collection.JavaConversions.asScalaBuffer

import org.apache.mahout.cf.taste.common.Weighting
import org.apache.mahout.cf.taste.eval.{RecommenderBuilder, DataModelBuilder}
import org.apache.mahout.cf.taste.impl.common.FastByIDMap
import org.apache.mahout.cf.taste.impl.eval.{GenericRecommenderIRStatsEvaluator, AverageAbsoluteDifferenceRecommenderEvaluator}
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.{ThresholdUserNeighborhood, NearestNUserNeighborhood}
import org.apache.mahout.cf.taste.impl.recommender.knn.{NonNegativeQuadraticOptimizer, KnnItemBasedRecommender}
import org.apache.mahout.cf.taste.impl.recommender.slopeone.{SlopeOneRecommender, MemoryDiffStorage}
import org.apache.mahout.cf.taste.impl.recommender.svd.{SVDRecommender, ALSWRFactorizer}
import org.apache.mahout.cf.taste.impl.recommender.{GenericUserBasedRecommender, GenericItemBasedRecommender, GenericBooleanPrefUserBasedRecommender}
import org.apache.mahout.cf.taste.impl.similarity.{TanimotoCoefficientSimilarity, PearsonCorrelationSimilarity, LogLikelihoodSimilarity, EuclideanDistanceSimilarity}
import org.apache.mahout.cf.taste.model.{PreferenceArray, DataModel}
import org.apache.mahout.cf.taste.recommender.Recommender
import org.apache.mahout.common.RandomUtils
import org.junit.Test

class RecommenderIntroTests {

  @Test def intro() = {
    val model = new FileDataModel(new File("data/intro.csv"))
    val sim = new PearsonCorrelationSimilarity(model)
    // find nearest 2
    val neighborhood = new NearestNUserNeighborhood(2, sim, model)
    val recommender = new GenericUserBasedRecommender(model, neighborhood, sim)
    // get 1 (:2) for user_id 1 (:1)
    val recommendations = recommender.recommend(1, 1)
    for (recommendation <- recommendations) {
      println(">>> recommendation: " + recommendation)
    }
  }
  
  @Test def evaluate() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
//    val builder = new RecommenderBuilder() {
//      override def buildRecommender(model : DataModel) : Recommender = {
//        val similarity = new PearsonCorrelationSimilarity(model)
//        val neighborhood = new NearestNUserNeighborhood(2, similarity, model)
//        new GenericUserBasedRecommender(model, neighborhood, similarity)
//      }
//    }
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        new SlopeOneRecommender(model)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> score=" + score)
  }
  
  @Test def evaluatePrecisionRecall() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new GenericRecommenderIRStatsEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new PearsonCorrelationSimilarity(model)
        val neighborhood = new NearestNUserNeighborhood(2, similarity, model)
        new GenericUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    // precision/recall at 2
    val stats = evaluator.evaluate(builder, null, model, null, 2, 
      GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)
    println(">>> precision = " + stats.getPrecision() + 
      ", recall = " + stats.getRecall())
  }
  
  @Test def evaluateWithBooleanDataModel() = {
    RandomUtils.useTestSeed()
    val model = new GenericBooleanPrefDataModel(
      GenericBooleanPrefDataModel.toDataMap(
      new FileDataModel(new File("data/intro.csv"))))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val recoBuilder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        // doesn't work because Pearson does not take boolean prefs
        // its meaningless and results in 0/0 == NaN
//        val similarity = new PearsonCorrelationSimilarity(model)
        val similarity = new LogLikelihoodSimilarity(model)
        val neighborhood = new NearestNUserNeighborhood(10, similarity, model)
        new GenericUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    val modelBuilder = new DataModelBuilder() {
      override def buildDataModel(trainingData : FastByIDMap[PreferenceArray]) : DataModel = {
        new GenericBooleanPrefDataModel(GenericBooleanPrefDataModel.toDataMap(trainingData))
      }
    }
    val score = evaluator.evaluate(recoBuilder, modelBuilder, model, 0.9, 1.0)
    println(">>> (bools) score =" + score)
  }

  @Test def evaluatePrecisionRecallWithBooleanPrefs() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new GenericRecommenderIRStatsEvaluator()
    val recoBuilder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new LogLikelihoodSimilarity(model)
        val neighborhood = new NearestNUserNeighborhood(10, similarity, model)
//        new GenericUserBasedRecommender(model, neighborhood, similarity)
        new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    val modelBuilder = new DataModelBuilder() {
      override def buildDataModel(trainingData : FastByIDMap[PreferenceArray]) : DataModel = {
        new GenericBooleanPrefDataModel(GenericBooleanPrefDataModel.toDataMap(trainingData))
      }
    }
    // precision/recall at 10
//    val stats = evaluator.evaluate(recoBuilder, modelBuilder, model, null, 
//      10, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0)
//    println(">>> (bools) precision = " + stats.getPrecision() + 
//      ", recall = " + stats.getRecall())
  }

    @Test def evaluateThresholdNeighborhood() = {
    RandomUtils.useTestSeed()
//    val model = new FileDataModel(new File("data/intro.csv"))
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new PearsonCorrelationSimilarity(model)
//        val neighborhood = new NearestNUserNeighborhood(2, similarity, model)
        val neighborhood = new ThresholdUserNeighborhood(0.7, similarity, model)
        new GenericUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (threshold neighborhood) score=" + score)
  }

  @Test def evaluateWeightedPearson() = {
    RandomUtils.useTestSeed()
//    val model = new FileDataModel(new File("data/intro.csv"))
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new PearsonCorrelationSimilarity(model, Weighting.WEIGHTED)
//        val neighborhood = new NearestNUserNeighborhood(2, similarity, model)
        val neighborhood = new ThresholdUserNeighborhood(0.7, similarity, model)
        new GenericUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (weighted pearson) score=" + score)
  }

  @Test def evaluateEucledianSimilarity() = {
    RandomUtils.useTestSeed()
//    val model = new FileDataModel(new File("data/intro.csv"))
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new EuclideanDistanceSimilarity(model)
//        val similarity = new PearsonCorrelationSimilarity(model, Weighting.WEIGHTED)
//        val neighborhood = new NearestNUserNeighborhood(2, similarity, model)
        val neighborhood = new ThresholdUserNeighborhood(0.7, similarity, model)
        new GenericUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (euclidean) score=" + score)
  }

  @Test def evaluateTanimoto() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new TanimotoCoefficientSimilarity(model)
        val neighborhood = new ThresholdUserNeighborhood(0.7, similarity, model)
        new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (tanimoto) score=" + score)
  }

  @Test def evaluateItemSimilarity() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new PearsonCorrelationSimilarity(model)
        new GenericItemBasedRecommender(model, similarity)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (item) score=" + score)
  }
  
  @Test def evaluateItemSlopeOne() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
//        new SlopeOneRecommender(model)
        // disable default weighting - results are worse
        val diffstorage = new MemoryDiffStorage(model, Weighting.UNWEIGHTED, Long.MaxValue)
        new SlopeOneRecommender(model, Weighting.UNWEIGHTED, Weighting.UNWEIGHTED, diffstorage)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (item/slope-1) score=" + score)
  }

  // experimental/slow
  
  @Test def evaluateItemSVD() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        // factorizer(_1)=#-topics, (_2)=lambda, (_3)=number_training_steps
        new SVDRecommender(model, new ALSWRFactorizer(model, 10, 0.5, 10))
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (item/SVD) score=" + score)
  }

  @Test def evaluateItemKnn() = {
    RandomUtils.useTestSeed()
    val model = new FileDataModel(new File("data/intro.csv"))
    val evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator()
    val builder = new RecommenderBuilder() {
      override def buildRecommender(model : DataModel) : Recommender = {
        val similarity = new LogLikelihoodSimilarity(model)
        val optimizer = new NonNegativeQuadraticOptimizer()
        // uses 10 nearest item neighborhood
        new KnnItemBasedRecommender(model, similarity, optimizer, 10)
      }
    }
    // eval 70 train/30 test split, use 100% users for eval
    val score = evaluator.evaluate(builder, null, model, 0.7, 1.0)
    println(">>> (item/Knn) score=" + score)
  }
}
