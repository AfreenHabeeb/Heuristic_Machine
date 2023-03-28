package com.verizon.HeuristicMachineModel

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.tree.CategoricalSplit
import org.apache.spark.ml.tree.ContinuousSplit
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.model.Node
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.count
import org.apache.spark.sql.functions.dense_rank
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions.regexp_extract
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.types.StringType
import org.slf4j.LoggerFactory
import java.text.SimpleDateFormat
import java.util.Calendar

object DTChat extends java.io.Serializable {

  val conf = new SparkConf().setAppName("DT_Chat_Rules")
  val sc = new SparkContext(conf)
  val spark = new HiveContext(sc);
  val logger = LoggerFactory.getLogger("DTChat")

  case class VariableImportance(featureName: String, importance: Double)
  case class OutputSchema(node: Int, 
                          prediction: Double, 
                          probability: Double, 
                          rules: String, 
                          coverage: Long, 
                          frequency: Long)

  case class DTRule(featureName: String, condition: String, threshold: Double)

  case class DTNode(id: Int,
                    isLeaf: Boolean,
                    prediction: Double,
                    probability: Double,
                    impurity: Double,
                    featureName: String,
                    threshold: Double,
                    featureType: String,
                    ruleList: List[DTRule])

  val nullNode = new DTNode(0, false, 0.0, 0.0, 0.0, "Empty", 0.0, "Empty", List[DTRule]())

  def main(args: Array[String]): Unit = {
    run(spark)
  }

  def run(spark: HiveContext) = {
    logger.info("START")
    val inputDF = spark.sql(query).na.drop

    val colMap = Map("first_hit_pagename" -> 15,
      "post_prop11_concat" -> 25)

    val getFirstProp11 = udf { (prop: String) =>
      val arr = prop.split("->").filter(_ != "")
      if (arr.isEmpty) ""
      else arr(0).replace("//", "|")
                 .replace("/", "|")
                 .replace("| |", "|")
                 .replace("||", "|")
    }

    val getLastPipePairRegex = "([\\s\\p{P}\\w<>=+]*\\|?[\\s\\p{P}\\w<>=+]+$)"
    val removeIllegalFirstCharRegex = "^[^a-zA-Z]*"

    val dfWithProp11 = inputDF.withColumn("post_prop11_concat", regexp_replace(
                                                                  regexp_extract(
                                                                    getFirstProp11(
                                                                         col("post_prop11_concat")), getLastPipePairRegex, 1), 
                                                                            removeIllegalFirstCharRegex, ""))
    val df = reduceColValues(colMap.toList, dfWithProp11)
    logger.info("Cols Reduced")

    df.cache
    logger.info("DF Cached")

    val responseColumn = "order_placed"
    val responseColumnIndex = responseColumn + "_index"
    val numImpColumns = 14
    
    val rfParams = getRFParams(responseColumnIndex)
    val randomForestDF = doRandomForest(df, responseColumn, rfParams)

    logger.info("RF Succesfully Run:")

    //getting the required stuff for decision tree
    val inputColumns = randomForestDF.select("featureName")
                                     .orderBy(desc("importance"))
                                     .map(_.getString(0))
                                     .take(numImpColumns) :+ responseColumn //replace 14 with variable
                                     
    val dfForDt = df.select(inputColumns.head, inputColumns.tail: _*)

    
    val dtParams = getDTParams(responseColumnIndex)
    val decisionTreeDF = doDecisionTree(dfForDt, responseColumn, dtParams)

    logger.info("DT Succesfully Run:")

    spark.sql("drop table if exists heuristic_machine_dev.chat_dt_rules")
    decisionTreeDF.write.saveAsTable("heuristic_machine_dev.chat_dt_rules")
  }
  
  def getRFParams(responseColumn: String) = {
    val featureColumn = "features_index"
    val rfImpurity = "gini"
    val rfMaxDepth = "10"
    val rfMaxBins = "60"
    val rfNumTrees = "200"

    Map("responseColumn" -> responseColumn,
      "featureColumn" -> featureColumn,
      "impurity" -> rfImpurity,
      "maxDepth" -> rfMaxDepth,
      "maxBins" -> rfMaxBins,
      "numTrees" -> rfNumTrees)
  }
  
  def getDTParams(responseColumn: String) = {
    val featureColumn = "features_index"
    val dtNumClasses = "2"
    val categoricalFeaturesInfo = Map[Int, Int]()
    val dtImpurity = "gini"
    val dtMaxDepth = "5"
    val dtMaxBins = "64"

    Map("responseColumn" -> responseColumn,
      "impurity" -> dtImpurity,
      "featureColumn" -> featureColumn,
      "numClasses" -> dtNumClasses,
      "maxDepth" -> dtMaxDepth,
      "maxBins" -> dtMaxBins)
  }

  val query = {
    """select css_sale_qualified,
           css_operating_system, 
           cast(order_placed as double),
           css_engagement_duration, 
           timeofday, 
           device_type, 
           day_of_week_for_breakouts,
           post_prop11_concat,
           regexp_extract(first_hit_pagename, "([\\s\\p{P}\\w<>=]*\\|?[\\s\\p{P}\\w]+$)",1) as first_hit_pagename
        from heuristic_machine_dev.funnel_collect_sep where serv_addr_id != 0 and css_b_business_rule_name like '%WLN_Con_Sales%'"""
  }

  def doRandomForest(df: DataFrame, responseColumn: String, rfParams: Map[String, String]): DataFrame = {
    logger.info("Prepping random forest")
    //prepping random forest
    //labelCol: String, featureCol: String, impurity: String, maxDepth: Int, maxBins: Int, numTrees: Int
    val rfPipelineStages = getPipelineStages(df, responseColumn) :+ getFeatureIndexer("features", "features_index") :+ getRandomForestClassifier(rfParams)

    logger.info("RF Pipeline built")
    val rfPiepline = new Pipeline().setStages(rfPipelineStages)

    logger.info("RF Pipeline created")
    return runRandomForest(df, rfPiepline)
  }

  def doDecisionTree(df: DataFrame, responseColumn: String, dtParams: Map[String, String]) : DataFrame = {
    logger.info("Prepping DT")

    val dtPipelineStages = getPipelineStages(df, responseColumn) :+ 
                           getFeatureIndexer("features", "features_index") :+ 
                           getDecisionTreeClassifier(dtParams)
                           
    logger.info("DT pipeline built")

    val dtPipeline = new Pipeline().setStages(dtPipelineStages)
    logger.info("DT pipeline created")

    val (model, predictionsDF) = runDecisionTree(df, dtPipeline, dtParams)
    logger.info("DT Sucesfully run")
    
    return getDecisionTreeModelAsDF(model, predictionsDF, dtPipeline)
  }

  def getDecisionTreeModelAsDF(model: DecisionTreeModel, predictionsDF: DataFrame, dtPipeline: Pipeline) = {
    val dtVectorAssembler = getVectorAssembler(dtPipeline)
    val featureColumns = getInputFeatureColumns(dtVectorAssembler)
    val nodeList = getNodesAsList(model.topNode, featureColumns, predictionsDF)
    val leafNodes = nodeList.filter(_.isLeaf)
    

    import spark.implicits._

    leafNodes.map(calculateLeafAttributes(_, predictionsDF)).toDF
  }

 /* def extractRulesFromModelTree(aNode: Node, left: String): String = {
    var treeoutput = ""
    var nodeValues = ""
    var predictionString = aNode.predict.toString.split(" ")
    var prediction = predictionString(0)
    var prob = predictionString(3).substring(0, predictionString(3).size - 1)
    nodeValues = aNode.id + "\t" + prediction + "\t" + prob + "\t" + aNode.impurity
    if (aNode.isLeaf) {
      rulestring = rulestring + "\n" + nodeValues
    } else {
      var leftsplitcondtion = ""
      var rightsplitcondtion = ""
      var nodeid = aNode.id
      var featuretype = aNode.split.get.featureType
      var featurenumber = aNode.split.get.feature
      var threshold = aNode.split.get.threshold
      var nodeSplitRuleCondition = ""
      if (featuretype.toString == "Continuous") {
        leftsplitcondtion = "feature " + featurenumber + " <= " + threshold
        rightsplitcondtion = "feature " + featurenumber + " > " + threshold
      } else {
        leftsplitcondtion = "feature " + featurenumber + " in " + threshold
        rightsplitcondtion = "feature " + featurenumber + " not in " + threshold
      }
      nodeSplitRuleCondition = leftsplitcondtion + "," + rightsplitcondtion
      nodeValues = nodeValues + "\t" + nodeSplitRuleCondition
      rulestring = rulestring + "\n" + nodeValues
      extractRulesFromModelTree(aNode.leftNode.get, "true")
      extractRulesFromModelTree(aNode.rightNode.get, "false")
    }
    return rulestring

  }*/

  /*
 * Convert the rules condition from feature vectors to the actual variable and its values.
 * Sample Input : "feature 7 > 0"
 * Sample Output: (custcbcatname in ("Upgrade_Orders/Others","Troubleshoot/Others","Bill/Others"))
 */
/*  def featureToColumnName(feature: String, df_lookup: Array[org.apache.spark.sql.DataFrame]): String = {
    val inputStringArray = feature.trim.split(" ", -1)
    val featureIndex = inputStringArray(1).toInt
    val logicalOperator = inputStringArray(2)
    val filterValues = inputStringArray(3)
    val featurename = featureColumnsArray(featureIndex)
    val filterCondition = "`" + featurename + "_index` " + logicalOperator + " " + filterValues
    var dfQueryOutput = df_lookup(featureIndex).filter(filterCondition).select(featurename).rdd.map(p => p.mkString).map(p => "\"" + p + "\"").collect().mkString(",")
    var transformString = ""
    if (dfQueryOutput == "") {
      transformString = "(`" + featurename + "` in " + "(" + "\"\"" + ")" + ")"
    } else {
      transformString = "(`" + featurename + "` in " + "(" + dfQueryOutput + ")" + ")"
    }
    return transformString
  }*/
  
  /**Construct a DT rule from the split values
   * */
  def constructDTRule(featureName: String, threshold: Double, direction: String = "left") = {
    val condition = if (direction == "left") " <= " else " > "
    DTRule(featureName, condition, threshold)
  }
  
  /**Form a SQL condition from the given DTRule.
   * Nominal rules are of the form - column in (values..)
   * Continuous rules are of the form - column <= threshold
   * */
  def getConditionFromRule(dtRule: DTRule, df: DataFrame) = {
    val featureName = dtRule.featureName
    val condition = dtRule.condition
    val threshold = dtRule.threshold
    val index = "_index"

    val isNominal = if (featureName.endsWith(index)) true else false
    val column = featureName.replace(index, "").toUpperCase
    
    if (isNominal) {
      

      column + " in (\"" + df.select(featureName, column)
        .where(featureName + condition + threshold)
        .select(column)
        .distinct
        .collect()
        .map(_.getString(0))
        .mkString("\",\"") + "\")"
    } else {
      column + condition + threshold
    }
  }

  def getNodeById(id: Int, nodes: ArrayBuffer[DTNode]) = {
    nodes.find(_.id == id).getOrElse[DTNode](nullNode)
  }
  
  /**Get the parent node for any given DT node in the list
   * */
  def getParentNode(nodes: ArrayBuffer[DTNode], currentRule: DTRule) = {
    val reversed = nodes.reverse
    def getParent(reverse: ArrayBuffer[DTNode]): ArrayBuffer[DTNode] = {
      val head = reverse.head
      if (!head.isLeaf && head.ruleList.head == currentRule) ArrayBuffer[DTNode](reverse.head)
      else getParent(reverse.tail)
    }
    getParent(reversed).head
  }
  
  def getOppositeCondition(condition: String) = {
    if (condition.trim == "<=") " > "
    else " <= "
  }
  
  /**Function to return a list of DTNodes.
   * Each DTNode represents a node in the tree.
   * Each DTNode also contains the list of rules which form leading up to forming that node.
   * */
  def getNodesAsList(node: Node,
                     featureColumns: Array[String],
                     df: DataFrame,
                     nodes: ArrayBuffer[DTNode] = ArrayBuffer[DTNode](),
                     ruleList: List[DTRule] = List[DTRule](),
                     direction: String = "left"): ArrayBuffer[DTNode] = {

    def extractNodeValues = {
      val id = node.id
      val isLeaf = node.isLeaf
      val prediction = node.predict.predict
      val probablility = node.predict.prob
      val impurity = node.impurity

      val (featureName, threshold, featureType, nodeRule) = if (!isLeaf) {
        //The rule for right direction is actually opposite of its left counterpart, which in tree language 
        //is its sibling. The sibling will now be in the head position in the rule list (since we are traversing
        //inorder)
        val newRule = if (direction == "right") {
          val sibling = ruleList.head
          DTRule(sibling.featureName, getOppositeCondition(sibling.condition), sibling.threshold) :: ruleList.tail
        } else ruleList

        val nodeSplit = node.split.get
        val featureName = featureColumns(nodeSplit.feature)
        val threshold = nodeSplit.threshold

        val nodeRule = constructDTRule(featureName, threshold) :: newRule

        (featureName, threshold, nodeSplit.featureType.toString, nodeRule)
      } else { //if node is leaf
        val parentNode = getParentNode(nodes, ruleList.head)
        ("Leaf", -99.99, "leaf", constructDTRule(parentNode.featureName, parentNode.threshold, direction) :: ruleList.tail)
      }

      DTNode(id, isLeaf, prediction, probablility, impurity, featureName, threshold, featureType, nodeRule)
    }

    val dtNode = extractNodeValues
    nodes += dtNode
    node.isLeaf match {
      case false =>
        getNodesAsList(node.leftNode.get, featureColumns, df, nodes, dtNode.ruleList, "left")
        getNodesAsList(node.rightNode.get, featureColumns, df, nodes, dtNode.ruleList, "right")
      case true => nodes
    }
  }


 /* def configureDTOutput(rootNode: Node, df: DataFrame, responseColumn: String, dtFit: PipelineModel): DataFrame = {

    import spark.implicits._

    featureColumnsArray = df.columns.filter(p => p != responseColumn)
    var df_lookup: Array[org.apache.spark.sql.DataFrame] = new Array[org.apache.spark.sql.DataFrame](featureColumnsArray.size)

    for (i <- 0 to featureColumnsArray.size - 1) {
      df_lookup(i) = dtFit.stages(i).transform(df).select(featureColumnsArray(i), featureColumnsArray(i) + "_index").distinct
    }

    if (rootNode.isLeaf) {
      print("ROOT NODE IS LEAF NODE!! HENCE IGNORING THIS INPUT !!")
      val outputDF = spark.createDataFrame(sc.emptyRDD[outputjoinschema])
      return outputDF
    } else {
      rulestring = ""
      modelTreeOutput = ""
      modelTreeOutput = extractRulesFromModelTree(rootNode, "true")
      var modelTreeOutputArray = modelTreeOutput.split("\n")
      modelTreeOutputArray.map(p => p.split("\t")).filter(p => p.size > 4).map(p => (rulemap += p(0).trim.toInt -> p(4)))
      val parentNodes = modelTreeOutputArray.map(p => p.split("\t")).filter(p => p.size > 4).distinct
      val leafNodes = modelTreeOutputArray.map(p => p.split("\t")).filter(p => p.size == 4).map(p => p(0)).distinct

      val allnodes = modelTreeOutputArray.map(p => p.split("\t")).filter(p => p.size > 2)
      val outputArrayRDD = sc.parallelize(allnodes)
      val outputdf1 = outputArrayRDD.map(p => dtreerules(p(0).toString, p(1).toString, p(2).toString)).toDF.distinct

      val outputArray = allnodes.map(p => Array(p(0), p(1), p(2), p(3), if (p(0) == "1") { "" } else { if ((p(0).toInt) % 2 == 0) { featureToColumnName(rulemap.get(p(0).toInt / 2).get.split(",")(0), df_lookup) } else { featureToColumnName(rulemap.get(p(0).toInt / 2).get.split(",")(1), df_lookup) } }))
      outputArray.map(p => (featureRulesmap += p(0).trim.toInt -> p(4).toString))

      // val outputArrayRDD=sc.parallelize(outputArray)

      //outputdf1.saveAsTable("heuristic_machine_dev.fractaloutputtable")
      // val nodeDetailsDf=outputArrayRDD.map(p =>dtreerules(p(0),p(1),p(2),p(3),p(4))).toDF
      //outputdf1.saveAsTable("fractaloutputtable222")

      val totalrecords = df.count().toInt
      finaloutput = ""
      spark.sql("drop table if exists heuristic_machine_dev.chat_dt_temp")
      df.write.mode("overwrite").saveAsTable("heuristic_machine_dev.chat_dt_temp")
      //inputdf.saveAsTable("heuristic_machine_dev.chat_dt_temp")
      spark.cacheTable("heuristic_machine_dev.chat_dt_temp")

      if (leafNodes.size > 3) {
        for (leafnode <- leafNodes) {
          leafnoderules = ""
          print("The value of leafnode in DT is: " + leafnode)
          var inputmap = scala.collection.mutable.Map[String, String]()
          var n = leafnode.toInt
          while (n != 1) {
            leafnoderules = featureRulesmap.get(n).get + " && " + leafnoderules
            n = n / 2
          }
          // leafnoderules.foreach(println)
          if (leafnoderules.size > 0) {
            val leafnoderulescondition = leafnoderules.substring(0, leafnoderules.size - 4)
            leafnoderulescondition.split(" && ").filter(p => (p.trim != null)).map(p => (p.split(" in ")(0), p)).map(p => (p._1.substring(1, p._1.length), p._2)).map(p => (inputmap += p._1 -> p._2))
            val rulesCondition = inputmap.toSeq.map(p => p._2).mkString(" and ")
            val conditionsForPrinting = inputmap.toSeq.map(p => p._2).mkString(" <br> ").replace("post_prop11_concat", "last_event_before_chat")
            val sqlcondition = "select count(*) from heuristic_machine_dev.chat_dt_temp  where " + rulesCondition
            val coverage = spark.sql(sqlcondition).rdd.map(p => p.toString()).collect()
            val coverageCount = coverage(0).substring(1, coverage(0).size - 1).toFloat
            val frequency = ((coverageCount * 100) / totalrecords)
            finaloutput = finaloutput + "----" + leafnode + "::" + conditionsForPrinting + "::" + coverageCount + "::" + frequency
          }
        }
        spark.uncacheTable("heuristic_machine_dev.chat_dt_temp")
        val outputrdd = sc.parallelize(finaloutput.split("----"))
        val splittedrdd = outputrdd.map(p => p.split("::", -1)).filter(p => p.size > 3)
        var outputdf = splittedrdd.map(p => NewOutputSchema(p(0), p(1).replaceAll("\"", "").replaceAll("`", ""), p(2), p(3))).toDF

        val outputjoinDf = outputdf.join(outputdf1, outputdf("node") === outputdf1("node")).select(outputdf("node"), outputdf1("prediction"), outputdf1("probability"), outputdf("rules"), outputdf("coverage"), outputdf("frequency"))

        //case class dtreerules(node: String, prediction: String, probability: String, impurity: String, rule: String)
        //  case class outputschema(node: String, rules: String, coverage: String, frequency: String)
        //outputdf = outputdf.join(nodeDetailsDf, outputdf("node") === nodeDetailsDf("node")).select("node","prediction","rules","coverage","frequency")

        spark.sql("DROP TABLE IF EXISTS  " + outputTableName)
    outputdf.saveAsTable(outputTableName)

        val opDFCount = outputjoinDf.count()
        //val opSample = outputdf.take(5)
        print(" #################################### \n ")
    print(" #################################### \n")
        print(" ###### The count of output DF in DT Logic is : " + opDFCount + " \n")

        return outputjoinDf
      } else {
        print("ROOT NODE HAS ONLY ONE LEAF NODE!! HENCE IGNORING THIS INPUT !!")
        val outputDF = spark.createDataFrame(sc.emptyRDD[outputjoinschema])
        return outputDF
      }
    }
  }*/
  
  /**Return an array of feature columns used
   * */
  def getInputFeatureColumns(vectorAssembler: VectorAssembler): Array[String] = {
    val inputColParam = vectorAssembler.getParam("inputCols")
    return vectorAssembler.extractParamMap.get(inputColParam).get.asInstanceOf[Array[String]]
  }

  /**
   * Get the vector assembler from the Pipeline
   */
  def getVectorAssembler(pipeline: Pipeline) = {
    pipeline.getStages.filter(_.isInstanceOf[VectorAssembler]).last.asInstanceOf[VectorAssembler]
  }

  def runDecisionTree(df: DataFrame, pipeline: Pipeline, dtParams: Map[String, String]) = {
    val dtFit = pipeline.fit(df)
    val predictionsDF = dtFit.transform(df)
    val categoricalFeaturesInfo = Map[Int, Int]()

    val responseColumn = dtParams("responseColumn")
    val numClasses = dtParams("numClasses").toInt
    val impurity = dtParams("impurity")
    val maxDepth = dtParams("maxDepth").toInt
    val maxBins = dtParams("maxBins").toInt

    val labeled = predictionsDF.map(row => LabeledPoint(row.getAs[Double](responseColumn), row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
    val model = DecisionTree.trainClassifier(labeled, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val rootNode = model.topNode
    
    //DecisionTreeClassificationModel
    val now = Calendar.getInstance().getTime()
    val dateFormat = new SimpleDateFormat("dd_hh_mm")
    val filename = dateFormat.format(now)
    
    logger.debug("****** MODEL ******")
    logger.debug(model.toDebugString)
    logger.debug("*******************")

    (model, predictionsDF)
  }

  def calculateLeafAttributes(leafNode: DTNode, predictionsDF: DataFrame) = {
    val extractedRule = leafNode.ruleList.map(getConditionFromRule(_, predictionsDF))
    val nominalQueries = extractedRule.filter(_.contains(" in ")).mkString(" and ")
    val continousQueries = extractedRule.filterNot(_.contains(" in ")).mkString(" and ")

    val conjucture = if (!nominalQueries.isEmpty && !continousQueries.isEmpty) " and " else ""
    val nominalQuery = nominalQueries.replace(" -> ", " in ")

    val query = nominalQuery + conjucture + continousQueries
    val totalCount = predictionsDF.count
    val coverage = predictionsDF.where(query).count

    val frequency = (coverage * 100) / totalCount
    
    val ruleToPrint = extractedRule.mkString("  &&  ").replace("(","[").replace(")","]").replace(" in "," : ")
    //val ruleToPrint = "(" + ruleList.mkString(") and (") + ")"
    //OutputSchema(node: Int, prediction: Double, probability: Double, rules: String, coverage: Long, frequency: Long)
    OutputSchema(leafNode.id, leafNode.prediction, leafNode.probability, ruleToPrint, coverage, frequency)
  }
  
  /**Run RF classifier and return a DF with the most important features.
   * */
  def runRandomForest(df: DataFrame, pipeline: Pipeline) = {
    val rfFit = pipeline.fit(df)
    val rfPredictionDF = rfFit.transform(df)

    //Get the metadata for the features column from predicted data frames and get the feature and the corresponding name by uisng the above declared function
    val featureMetaAttributes = rfPredictionDF.schema("features")
      .metadata
      .getMetadata("ml_attr")
      .getMetadata("attrs")

    //Declaring a function which get the fearture number and the corresponding name from the metadata object
    val f: (Metadata) => (Long, String) = { m => (m.getLong("idx"), m.getString("name")) }

    val numericFeatures = if (featureMetaAttributes.contains("numeric")) featureMetaAttributes.getMetadataArray("numeric").map(f).sortBy(_._1) else Array.empty[(Long, String)]
    val nominalFeatues = if (featureMetaAttributes.contains("nominal")) featureMetaAttributes.getMetadataArray("nominal").map(f).sortBy(_._1) else Array.empty[(Long, String)]

    val features = numericFeatures ++ nominalFeatues

    val rfModel = rfFit.stages(pipeline.getStages.size - 1).asInstanceOf[RandomForestClassificationModel]
    //Joining the Importance vector array with the features lookup from the above to get the feature importance value
    val fImportance = rfModel.featureImportances.toArray.zip(features)
      .map(x => (x._2._2, x._1))
      .sortBy(-_._2)

    import spark.implicits._

    sc.parallelize(fImportance).map(p => VariableImportance(p._1.replace("_index", ""), p._2)).toDF
  }
  
   /**Build aand return RandomForestClassifier
   * */
  def getRandomForestClassifier(params: Map[String, String]): RandomForestClassifier = {
    return new RandomForestClassifier()
      .setLabelCol(params("responseColumn"))
      .setFeaturesCol(params("featureColumn"))
      .setImpurity(params("impurity"))
      .setMaxDepth(params("maxDepth").toInt)
      .setMaxBins(params("maxBins").toInt)
      .setNumTrees(params("numTrees").toInt)
  }
  
  /**Build a pipeline with a StringIndexer and FeatureVectorAssembler
   * */
  def getPipelineStages(df: DataFrame, responseColumn: String) = {

    val categorialColumns = df.schema.filter(_.dataType == StringType).map(_.name).toArray
    val numericalColumns = df.columns diff categorialColumns
    val index = "_index"
    val features = "features"

    //Converting the string labels to label indices
    val stringIndexer = (categorialColumns :+ responseColumn).map(cname => new StringIndexer()
      .setInputCol(cname)
      .setOutputCol(cname + index))

    //Creating a vector assembler to create a single feature column and converting into indexed features
    //Will be excluding the response columns
    val featureColumns = numericalColumns.filter(_ != responseColumn) ++ categorialColumns.filter(_ != responseColumn).map(_ + index)
    val featureAssembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol(features)

    stringIndexer :+ featureAssembler //:+ featureIndexer
  }
  
   /**Build aand return VectorIndexer
   * */
  def getFeatureIndexer(inputCol: String, outputCol: String): VectorIndexer = {
    return new VectorIndexer().setInputCol(inputCol).setOutputCol(outputCol)
  }
  
 
  
  /**Build aand return DecisionTreeClassifier
   * */
  def getDecisionTreeClassifier(params: Map[String, String]): DecisionTreeClassifier = {
    return new DecisionTreeClassifier()
      .setLabelCol(params("responseColumn"))
      .setFeaturesCol(params("featureColumn"))
      .setImpurity(params("impurity"))
      .setMaxDepth(params("maxDepth").toInt)
      .setMaxBins(params("maxBins").toInt)
  }
  
  /**Take topN counts from each column and label all the other column values as 'Other'.
   * Also renames blank values as 'Unknown'
   * */
  def reduceCol(df: DataFrame, colName: String, topN: Int): DataFrame = {
    return df.withColumn("count", count(colName) over Window.partitionBy(colName))
      .withColumn("rank", dense_rank.over(Window.orderBy(desc("count"), desc(colName))))
      .withColumn(colName, when(col("rank") > topN, "Other")
        .when(col(colName).equalTo(""), "Unknown")
        .otherwise(col(colName)))
      .drop("count").drop("rank")
  }
  
  
  /**Function to apply column reduction on eligible column in the DF recursively.
   * The columns in the colMapList are the ones to reduce.
   * This function calls reduceCol which does the actual dirty work
   * 
   * */
  @tailrec
  def reduceColValues(colMapList: List[(String, Int)], df: DataFrame): DataFrame = colMapList match {
    case Nil       => df
    case l :: rest => reduceColValues(rest, reduceCol(df, l._1, l._2))
  }

  def getSplit(split: org.apache.spark.ml.tree.Split) = split match {
    case s: CategoricalSplit => s.leftCategories
    case s: ContinuousSplit  => s.threshold
  }
}