#!/bin/bash

SCALA_HOME=/opt/scala-2.9.2
MAHOUT_HOME=/opt/mahout-distribution-0.7

mkdir -p target/fatjar
cd target/fatjar
jar xvf $MAHOUT_HOME/core/target/mahout-core-0.7-job.jar
jar xvf $SCALA_HOME/lib/scala-library.jar
jar xvf ../scala-2.9.2/mia-scala-examples_2.9.2-1.0.jar
jar cvf /tmp/my-mahout-fatjar.jar *
cd -
rm -rf target/fatjar
