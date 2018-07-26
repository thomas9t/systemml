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
import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json._
import java.util.concurrent.atomic.LongAdder
import scala.concurrent.{Await, Future, future}

// format: can be file, binary, csv, ijv, jpeg, ...
case class PredictionRequest(model:String, inputs:String, format:String, num_input:Option[Int])
case class Model(model:String, path:String)
case class PredictionResponse(response:String, format:String)

trait PredictionJsonProtocol extends SprayJsonSupport  with DefaultJsonProtocol {
  implicit val predictionRequestFormat = jsonFormat4(PredictionRequest)
  implicit val predictionResponseFormat = jsonFormat2(PredictionResponse)
}

class PredictionService {
  
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
wget http://central.maven.org/maven2/com/typesafe/akka/akka-http-spray-json-experimental_2.11/2.4.11.2/akka-http-spray-json-experimental_2.11-2.4.11.2.jar
wget http://central.maven.org/maven2/io/spray/spray-json_2.11/1.3.2/spray-json_2.11-1.3.2.jar
JARS=$JARS"config-1.3.3.jar:ssl-config-core_2.11-0.2.4.jar:reactive-streams-1.0.2.jar:commons-cli-1.4.jar:scala-parser-combinators-2.11.0-M4.jar:scala-library-2.11.12.jar:akka-http-spray-json-experimental_2.11-2.4.11.2.jar:spray-json_2.11-1.3.2.jar"
echo "Include the following jars into the classpath: "$JARS


2. Copy SystemML.jar and systemml-1.2.0-SNAPSHOT-extra.jar into the directory where akka jars are placed

3. Start the server:
java -cp $JARS org.apache.sysml.api.ml.serving.PredictionService -port 9000 -admin_password admin

4. Check the health of the server:
curl -u admin -XGET localhost:9000/health

5. Perform prediction
curl -XPOST -H "Content-Type:application/json" -d '{ "inputs":"1,2,3", "format":"csv", "model":"test", "num_input":1 }' localhost:9000/predict

6. Shutdown the server:
curl -u admin -XGET localhost:9000/shutdown

 */
object PredictionService extends PredictionJsonProtocol {
  // val LOG = LogFactory.getLog(classOf[PredictionService].getName())
  implicit val system = ActorSystem("systemml-prediction-service")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout = akka.util.Timeout(10 seconds)
  val userPassword = new HashMap[String, String]()
  var bindingFuture: Future[Http.ServerBinding] = null
  var scheduler:Scheduler = null
  
  def getCommandLineOptions():org.apache.commons.cli.Options = {
    val hostOption = new org.apache.commons.cli.Option("ip", true, "IP address")
    val portOption = new org.apache.commons.cli.Option("port", true, "Port number")
    val numRequestOption = new org.apache.commons.cli.Option("max_requests", true, "Maximum number of requests")
    val timeoutOption = new org.apache.commons.cli.Option("timeout", true, "Timeout in milliseconds")
    val passwdOption = new org.apache.commons.cli.Option("admin_password", true, "Admin password. Default: admin")
    val helpOption = new org.apache.commons.cli.Option("help", false, "Show usage message")
    val maxSizeOption = new org.apache.commons.cli.Option("max_bytes", true, "Maximum size of request in bytes")
    
    // Only port is required option
    portOption.setRequired(true)
    
    return new org.apache.commons.cli.Options()
          .addOption(hostOption).addOption(portOption).addOption(numRequestOption)
          .addOption(passwdOption).addOption(timeoutOption).addOption(helpOption).addOption(maxSizeOption)
  }
  
  def main(args: Array[String]): Unit = {
    // Parse commandline variables:
    val options = getCommandLineOptions
		val line = new PosixParser().parse(getCommandLineOptions, args);
		if(line.hasOption("help")) {
				new org.apache.commons.cli.HelpFormatter().printHelp( "systemml-prediction-service", options )
				return
		}
    userPassword.put("admin", line.getOptionValue("admin_password", "admin"))
    val currNumRequests = new LongAdder
    val maxNumRequests = if(line.hasOption("max_requests")) line.getOptionValue("max_requests").toLong else Long.MaxValue
    val timeout = if(line.hasOption("timeout"))  Duration(line.getOptionValue("timeout").toLong, MILLISECONDS) else Duration.Inf 
    val sizeDirective = if(line.hasOption("max_bytes")) withSizeLimit(line.getOptionValue("max_bytes").toLong) else withoutSizeLimit 
    
    // Initialize statistics counters
    val numTimeouts = new LongAdder
    val numFailures = new LongAdder
    val totalTime = new LongAdder
    val numCompletedPredictions = new LongAdder
    
    // For now the models need to be loaded every time. TODO: pass the local to serialized models via commandline 
    val models = new HashMap[String, Model]
    models.put("test", new Model("test", "test-path"))
    
    // TODO: Set the scheduler using factory
    scheduler = new NoBatching(timeout)
    scheduler.addModel(models.get("test"))
    val gpus = null
    val numCores = 1
    val maxMemory = Runtime.getRuntime().totalMemory()
    scheduler.start(numCores, maxMemory, gpus)
    
    // Define unsecured routes: /predict and /health
    val unsecuredRoutes = {
      path("predict") {
        post {
          validate(currNumRequests.longValue() < maxNumRequests, "The prediction server received too many requests. Ignoring the current request.") {
            entity(as[PredictionRequest]) { request =>
              validate(models.containsKey(request.model), "The model is not available.") {
                try {
                  currNumRequests.increment()
                  val start = System.nanoTime()
                  val response = Await.result(scheduler.enqueue(request, models.get(request.model)), timeout)
                  totalTime.add(System.nanoTime()-start)
                  numCompletedPredictions.increment()
                  complete(StatusCodes.OK, response)
                } catch {
                  case e:scala.concurrent.TimeoutException => {
                    numTimeouts.increment()
                    complete(StatusCodes.RequestTimeout, "Timeout occured")
                  }
                  case e:Exception => {
                    numFailures.increment()
                    e.printStackTrace()
                    complete(StatusCodes.InternalServerError, "Exception occured while executing the prediction request:" + e.getMessage)  
                  }
                } finally {
                  currNumRequests.decrement()
                }
              }
            }
          }
        }
      }
    }
    
    // For administration: This can be later extended for supporting multiple users.
    val securedRoutes = {
      authenticateBasicAsync(realm = "secure site", userAuthenticate) {
        user =>
          path("shutdown") {
            get {
              shutdownService(user, scheduler)
            }
          } ~
          path("health") {
            get {
              val stats = "Number of requests (total/completed/timeout/failures):" + currNumRequests.longValue() + "/" + numCompletedPredictions.longValue() + "/"
                  numTimeouts.longValue() + "/" + numFailures.longValue() + ".\n" +
                  "Average prediction time:" + ((totalTime.doubleValue()*1e-6) / numCompletedPredictions.longValue()) + " ms.\n" 
              complete(StatusCodes.OK, stats)
            }
          }
      }
    }
    
    bindingFuture = Http().bindAndHandle(
        sizeDirective { // Both secured and unsecured routes need to respect the size restriction
          unsecuredRoutes ~ securedRoutes
        }, 
      line.getOptionValue("ip", "localhost"), line.getOptionValue("port").toInt)
    
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
  
  def shutdownService(user:String, scheduler:Scheduler):StandardRoute = {
    if(user.equals("admin")) {
      try {
        Http().shutdownAllConnectionPools() andThen { case _ => bindingFuture.flatMap(_.unbind()).onComplete(_ ⇒ system.terminate()) }
        scheduler.shutdown()
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