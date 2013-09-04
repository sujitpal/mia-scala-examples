name := "mia-scala-examples"

EclipseKeys.withSource := true

version := "1.0"

scalaVersion := "2.9.2"

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-core" % "1.2.1",
  "org.apache.mahout" % "mahout-core" % "0.8",
  "nz.ac.waikato.cms.weka" % "weka-dev" % "3.7.6",
  "nz.ac.waikato.cms.weka" % "LibLINEAR" % "1.0.2",
  "de.bwaldvogel" % "liblinear" % "1.92",
  "org.apache.lucene" % "lucene-core" % "4.2.1",
  "org.apache.lucene" % "lucene-analyzers-common" % "4.2.1",
  "org.apache.lucene" % "lucene-queries" % "4.2.1",
  "org.apache.lucene" % "lucene-queryparser" % "4.2.1",
  "com.novocode" % "junit-interface" % "0.10" % "test"
)
