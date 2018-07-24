/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.api.ml.serving

import scala.concurrent.{ExecutionContext, Future}
import akka.http.scaladsl.server.{Route, StandardRoute}
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.model.{HttpEntity, StatusCodes}
import akka.http.scaladsl.Http
import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import com.typesafe.config.ConfigFactory
import org.apache.commons.cli.{PosixParser, CommandLineParser, CommandLine}
import scala.concurrent.duration._
import java.util.HashMap

case class PredictionRequest(modelId:String, input:String)
class PredictionService(implicit val executionContext: ExecutionContext) {
}

/*
Usage:
1. Compiling a fat jar with maven assembly plugin in our standalone jar created lot of issues. 
Hence, for time being, we recommend downloading jar using the below script:
SCALA_VERSION="2.11"
AKKA_HTTP_VERSION="10.1.3"
AKKA_VERSION="2.5.14"
PREFIX="http://central.maven.org/maven2/com/typesafe/akka/"
JARS=""
for PKG in actor stream protobuf
do
  PKG_NAME="akka-"$PKG"_"$SCALA_VERSION
  JAR_FILE=$PKG_NAME"-"$AKKA_VERSION".jar"
  wget $PREFIX$PKG_NAME"/"$AKKA_VERSION"/"$JAR_FILE
  JARS=$JARS$JAR_FILE":"
done
for PKG in http http-core parsing
do
  PKG_NAME="akka-"$PKG"_"$SCALA_VERSION
  JAR_FILE=$PKG_NAME"-"$AKKA_HTTP_VERSION".jar"
  wget $PREFIX$PKG_NAME"/"$AKKA_HTTP_VERSION"/"$JAR_FILE
  JARS=$JARS$JAR_FILE":"
done
wget http://central.maven.org/maven2/com/typesafe/config/1.3.3/config-1.3.3.jar
wget http://central.maven.org/maven2/com/typesafe/ssl-config-core_2.11/0.2.4/ssl-config-core_2.11-0.2.4.jar
wget http://central.maven.org/maven2/org/reactivestreams/reactive-streams/1.0.2/reactive-streams-1.0.2.jar
wget http://central.maven.org/maven2/org/scala-lang/scala-library/2.11.12/scala-library-2.11.12.jar
wget http://central.maven.org/maven2/org/scala-lang/scala-parser-combinators/2.11.0-M4/scala-parser-combinators-2.11.0-M4.jar
wget http://central.maven.org/maven2/commons-cli/commons-cli/1.4/commons-cli-1.4.jar
JARS=$JARS"config-1.3.3.jar:ssl-config-core_2.11-0.2.4.jar:reactive-streams-1.0.2.jar:commons-cli-1.4.jar:scala-parser-combinators-2.11.0-M4.jar:scala-library-2.11.12.jar"
echo "Include the following jars into the classpath: "$JARS


2. Copy SystemML.jar and systemml-1.2.0-SNAPSHOT-extra.jar into the directory where akka jars are placed

3. Start the server:
java -cp $JARS org.apache.sysml.api.ml.serving.PredictionService -port 9000 -admin_password admin

4. Check the health of the server:
curl -XGET localhost:9000/health

5. Shutdown the server:
curl -u admin -XGET localhost:9000/shutdown

 */
object PredictionService  {
  implicit val system = ActorSystem("systemml-prediction-service")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout = akka.util.Timeout(10 seconds)
  val userPassword = new HashMap[String, String]()
  var bindingFuture: Future[Http.ServerBinding] = null
  def main(args: Array[String]): Unit = {
    val hostOption = new org.apache.commons.cli.Option("ip", true, "IP address")
    val portOption = new org.apache.commons.cli.Option("port", true, "Port number")
    val passwdOption = new org.apache.commons.cli.Option("admin_password", true, "Admin password. Default: admin")
    portOption.setRequired(true)
    val options = new org.apache.commons.cli.Options().addOption(portOption)
		val line = new PosixParser().parse(options, args);
    userPassword.put("admin", line.getOptionValue("admin_password", "admin"))
    val unsecuredRoutes = {
      path("health") {
        get {
          complete(StatusCodes.OK, "Service is working great!")
        }
      }
    }
    
    // For administration: This can be later extended for supporting multiple users.
    val securedRoutes = {
      logRequestResult("akka-http-secured-service") {
        authenticateBasicAsync(realm = "secure site", userAuthenticate) {
          user =>
            path("shutdown") {
              get {
                shutdownService(user)
              }
            }
        }
      }
    }
    
    bindingFuture = Http().bindAndHandle(unsecuredRoutes ~ securedRoutes, line.getOptionValue("ip", "localhost"), line.getOptionValue("port").toInt)
    println(s"Prediction Server online.\nPress RETURN to stop...")
    scala.io.StdIn.readLine()
    bindingFuture
      .flatMap(_.unbind())
      .onComplete(_ ⇒ system.terminate())
  }
  
  def userAuthenticate(credentials: akka.http.scaladsl.server.directives.Credentials): Future[Option[String]] = {
    credentials match {
      case p @ akka.http.scaladsl.server.directives.Credentials.Provided(id) =>
        Future {
          if(userPassword.containsKey(id) && p.verify(userPassword.get(id))) Some(id)
          else None
        }
      case _ => Future.successful(None)
    }
  }
  
  def shutdownService(user:String):StandardRoute = {
    if(user.equals("admin")) {
      try {
        Http().shutdownAllConnectionPools() andThen { case _ => bindingFuture.flatMap(_.unbind()).onComplete(_ ⇒ system.terminate()) }
        complete(StatusCodes.OK, "Shutting down the server.")
      } finally {
        new Thread(new Runnable { 
          def run() {
            Thread.sleep(100) // wait for 100ms to send reply and then kill the prediction JVM so that we don't wait scala.io.StdIn.readLine()
            System.exit(0)
          }
        }).start();
      }
    }
    else {
      complete(StatusCodes.BadRequest, "Only admin can shutdown the service.")
    }
  }
  
}