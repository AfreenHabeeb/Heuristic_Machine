package com.verizon.HeuristicMachineModel

import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql._

import org.apache.spark.sql.types.Metadata
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext

/**
 * @author ${user.name}
 */
object RandomForestModel {
  case class VariableImportance(feature_name: String, importance: Double)

  
  def  main(args : Array[String]) {
  
  val conf = new SparkConf().setAppName("RF Model")
  val sc = new SparkContext(conf)
  val sqlContext= new HiveContext(sc);
  
  import sqlContext.implicits._
// BASE DF
 // val inputdf = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter",",").load("/etl/rfInput")


var tablename=args(0)
var selectedColumns=args(1)
var responseColumn=args(2)
var outputTableName=args(3)

// Setting the configuration parameters for RandomForestClassifier
val impurity = "gini"
val maxDepth = 5
val maxBins =60
val numTrees =200


//Columns names with special charactes like "-",spaces wont work which has to be escaped with backtic i.e "`". So adding the backtick for all columns
var normalizedColumnNames=selectedColumns.split(",").map(p => "`"+p+"`").mkString(",")
var sqlStatement="select "+normalizedColumnNames+" from "+tablename
val inputDF=sqlContext.sql(sqlStatement).na.drop

//val inputDatasetSplits=inputDF.randomSplit(Array(0.6, 0.4), seed = 11L)
//val (trainingDF, testDF) = (inputDatasetSplits(0), inputDatasetSplits(1))


//Converting the string labels to label indices
val columnArray=selectedColumns.split(",")
val index_transformers: Array[org.apache.spark.ml.PipelineStage] = columnArray.map(cname => new StringIndexer().setInputCol(cname).setOutputCol(s"${cname}_index"))

//Creating a vector assembler to create a single feature column and converting into indexed features except the response columns. Need to check if it is needed.
val featureColumns=columnArray.filter(p => p!=responseColumn).map(p => p+"_index")
val features_assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features")
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("features_index").setMaxCategories(10)

//Building RF Model
val rf = new RandomForestClassifier().setLabelCol(responseColumn+"_index").setFeaturesCol("features_index").setNumTrees(numTrees).setMaxBins(maxBins).setImpurity(impurity)

//Adding the pipeline stages i.e feature indexers, feature assembler, featureIndexer, RandomForest Model
val pipelinestages=index_transformers :+ features_assembler :+featureIndexer :+rf

val index_pipeline = new Pipeline().setStages(pipelinestages)

val index_model = index_pipeline.fit(inputDF)
val predictionsDF = index_model.transform(inputDF)


val rfModel = index_model.stages(pipelinestages.size-1).asInstanceOf[RandomForestClassificationModel]
var importanceVector = rfModel.featureImportances

//Declaring a function which get the fearture number and the corresponding name from the metadata object
val f: (Metadata) => (Long,String) = (m => (m.getLong("idx"), m.getString("name")))

//Get the metadata for the features column from predicted data frames and get the feature and the corresponding name by uisng the above declared function
val features= predictionsDF.schema("features").metadata.getMetadata("ml_attr").getMetadata("attrs").getMetadataArray("nominal").map(f).sortBy(_._1)

//Joining the Importance vector array with the features lookup from the above to get the feature importance value
val fImportance = importanceVector.toArray.zip(features).map(x=>(x._2._2,x._1)).sortBy(-_._2)

//Creating a hive table for the variable importance
var featureImportanceRDD=sc.parallelize(fImportance).map(p => Array(p._1,p._2))
var df=featureImportanceRDD.map(p => VariableImportance(p(0).toString.split("_index")(0),p(1).asInstanceOf[Double])).toDF()
sqlContext.sql("DROP TABLE IF EXISTS  "+outputTableName)
//df.saveAsTable(outputTableName)


  }

}
