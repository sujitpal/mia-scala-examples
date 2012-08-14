import sbt._
import sbt.Keys._

object MiascalaexamplesBuild extends Build {

  lazy val miascalaexamples = Project(
    id = "mia-scala-examples",
    base = file("."),
    settings = Project.defaultSettings ++ Seq(
      name := "mia-scala-examples",
      organization := "com.mycompany",
      version := "0.1-SNAPSHOT",
      scalaVersion := "2.9.2"
      // add other settings here
    )
  )
}
