package com.mycompany.mia.fpg

import java.io.{File, FileWriter, PrintWriter}

import scala.Array.canBuildFrom
import scala.actors.threadpool.AtomicInteger

import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.store.NIOFSDirectory

/**
 * Reads data from a Lucene index and writes out in format
 * expected by Mahout's FP Growth driver. Concept Map for
 * a document is stored in sparse representation as below:
 *   7996649$2.71 8002896$6.93 8256842$2.71 ...
 * The code below will convert it to the format expected by
 * Mahout's FP Growth algorithm as below:
 *   1  7996649 8002896 8256842 ...
 */
object Preprocess extends App {

  val IndexPath = "/path/to/lucene/index"
  val OutputFile = "data/imuids_p.csv"

  val reader = DirectoryReader.open(
    new NIOFSDirectory(new File(IndexPath)))
  val writer = new PrintWriter(new FileWriter(new File(OutputFile)))
  
  val ndocs = reader.numDocs()
  val counter = new AtomicInteger(1)
  (0 until ndocs).foreach(docId => {
    val recordId = counter.getAndIncrement()
    if (recordId % 1000 == 0)
      Console.println("Processed %d docs".format(recordId))
     val doc = reader.document(docId)
     val field = doc.getField("imuids_p")
     if (field != null) {
       val imuids = field.stringValue().split(" ").
         map(pair => pair.split("\\$")(0)).
         mkString(" ")
       writer.println("%d\t%s".format(recordId, imuids))
     }
  })
  writer.flush()
  writer.close()
  reader.close()
}
