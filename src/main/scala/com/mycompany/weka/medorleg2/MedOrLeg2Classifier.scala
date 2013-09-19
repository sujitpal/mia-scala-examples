package com.mycompany.weka.medorleg2

import java.io.{FileInputStream, ObjectInputStream}
import scala.Array.canBuildFrom
import weka.classifiers.functions.LibLINEAR
import weka.core.{Attribute, Instances, SparseInstance}
import weka.core.converters.ConverterUtils.DataSource

object MedOrLeg2Classifier extends App {

  val TrainARFFPath = "/path/to/training/ARFF/file" 
  val ModelPath = "/path/to/trained/WEKA/model/file"
  // copied from sklearn/feature_extraction/stop_words.py
  val EnglishStopWords = Set[String](
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves")
    
  val source = new DataSource(TrainARFFPath)
  val data = source.getDataSet()
  val numAttributes = data.numAttributes()
  data.setClassIndex(numAttributes - 1)
  
  // features: this is only necessary for trying to classify
  // sentences outside the training set (see last block). In
  // such a case we would probably store the attributes in 
  // some external datasource such as a database table or file.
  var atts = new java.util.ArrayList[Attribute]()
  (0 until numAttributes).foreach(j =>
    atts.add(data.attribute(j)))
  val vocab = Map[String,Int]() ++ 
    (0 until numAttributes - 1).
    map(j => (data.attribute(j).name(), j))
  
  // load model
  val modelIn = new ObjectInputStream(new FileInputStream(ModelPath))
  val model = modelIn.readObject().asInstanceOf[LibLINEAR]
  
  // predict using data from test set and compute accuracy
  var numCorrectlyPredicted = 0
  (0 until data.numInstances()).foreach(i => {
    val instance = data.instance(i)
    val expectedLabel = instance.value(numAttributes - 1).intValue()
    val predictedLabel = model.classifyInstance(instance).intValue()
    if (expectedLabel == predictedLabel) numCorrectlyPredicted += 1
  })
  Console.println("# instances tested: " + data.numInstances())
  Console.println("# correctly predicted: " + numCorrectlyPredicted)
  Console.println("Accuracy (%) = " + 
    (100.0F * numCorrectlyPredicted / data.numInstances()))
    
  // predict class of random sentences
  val sentences = Array[String](
    "Throughout recorded history, humans have taken a variety of steps to control family size: before conception by delaying marriage or through abstinence or contraception; or after the birth by infanticide.",
    "I certify that the preceding sixty-nine (69) numbered paragraphs are a true copy of the Reasons for Judgment herein of the Honourable Justice Barker.")
  sentences.foreach(sentence => {
    val indices = sentence.split(" ").
      map(word => word.toLowerCase()).
      map(word => word.filter(c => Character.isLetter(c))).
      filter(word => word.length() > 1).
      filter(word => !EnglishStopWords.contains(word)).
      map(word => if (vocab.contains(word)) vocab(word) else -1).
      filter(index => index > -1).
      toList
    val scores = indices.groupBy(index => index).
      map(kv => (kv._1, kv._2.size))
    val norm = math.sqrt(scores.map(score => score._2).
      foldLeft(0D)(math.pow(_, 2) + math.pow(_, 2)))
    val normScores = scores.map(kv => (kv._1, kv._2 / norm))
    val instance = new SparseInstance(numAttributes)
    normScores.foreach(score => 
      instance.setValue(score._1, score._2))
    val instances = new Instances("medorleg2_test", atts, 0)
    instances.add(instance)
    instances.setClassIndex(numAttributes - 1)
    val label = model.classifyInstance(instances.firstInstance()).toInt
    Console.println(label)
  })
}

