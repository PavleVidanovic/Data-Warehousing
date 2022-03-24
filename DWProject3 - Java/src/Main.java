import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.hive.ql.exec.spark.session.SparkSession;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;

import scala.Tuple2;

public class Main {

	
	
	public static void main(String[] args) {
		
		
		
		try {
			SparkConf conf = new SparkConf().setAppName("Main").setMaster("local[2]").set("spark.executor.memory", "4g")
					.set("spark.driver.memory", "4g");
			JavaSparkContext sc = new JavaSparkContext(conf);
			Logger.getLogger("org").setLevel(Level.OFF);
			Logger.getLogger("akka").setLevel(Level.OFF);

			
			String dataFile = "resource/graduate_admission_edited.txt";	
			JavaRDD<String> data = sc.textFile(dataFile);

			Map<String, Double> map = new HashMap<String, Double>();
			map.put("Yes", 0.0);
			map.put("Depends", 1.0);
			map.put("No", 2.0);

			JavaRDD<LabeledPoint> parsedData = data.map(line -> {
				String[] parts = line.split(",");
				double[] v = new double[parts.length - 1];
				for (int i = 0; i < parts.length - 1; i++) {
					v[i] = Double.parseDouble(parts[i]);
				}
				LabeledPoint l = new LabeledPoint(map.get(parts[parts.length - 1]), Vectors.dense(v));
				return l;
			});
			//do ovde je isti kod za svaki od klasifikatora
			

			//RANDOM FOREST
			JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[] { 0.73, 0.27 }, 11L);	//ne znam sta znaci 11L
			JavaRDD<LabeledPoint> trainingData = splits[0].cache();
			JavaRDD<LabeledPoint> testData = splits[1];

			
			//metrike koje se delimicno poklapaju sa Random Forest u weki gde smo dobili najbolje rezultate
			int numClasses = 3;
			Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
			int numTrees = 100;	
			String featureSubsetStrategy = "auto";
			String impurity = "gini";
			int maxDepth = 5;
			int maxBins = 32;
			int seed = 12345678;
			
			
			
			RandomForestModel model = RandomForest.trainClassifier(trainingData, 
					numClasses, categoricalFeaturesInfo,
					numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

			JavaPairRDD<Object, Object> predictionAndLabels = testData
					.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
			MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

			ClassificationMetrics.saveMetricsToFile(metrics, "metrics/RandomForest");

			model.save(sc.sc(), "models/RandomForest");

			RandomForestModel sameModel = RandomForestModel.load(sc.sc(), "models/RandomForest");
			
			//newData ima parametre kao poslednji podatak u nasem datasetu i on sluzi za testiranje naseg klasifikatora
			//trebalo bi da daje predikciju Yes, odnosno 0.0
			Vector newData = Vectors.dense(new double[] { 400, 333 ,117, 4, 5, 4, 9.66, 1});
			
			double prediction = sameModel.predict(newData);
			
			System.out.println("Model Prediction for Random Forest on New Data = " + prediction);
			
			//KRAJ RANDOM FOREST
			
			
			
			//NAIVE BAYES
			JavaRDD<LabeledPoint>[] splitsNaive = parsedData.randomSplit(new double[] { 0.72, 0.28 }, 11L);
			JavaRDD<LabeledPoint> trainingDataNaive = splitsNaive[0].cache();
			JavaRDD<LabeledPoint> testDataNaive = splitsNaive[1];

			NaiveBayesModel modelNaive = NaiveBayes.train(trainingDataNaive.rdd(), 1.0);

			JavaPairRDD<Object, Object> predictionAndLabelsNaive = testDataNaive
					.mapToPair(p -> new Tuple2<>(modelNaive.predict(p.features()), p.label()));
			MulticlassMetrics metricsNaive = new MulticlassMetrics(predictionAndLabelsNaive.rdd());

			ClassificationMetrics.saveMetricsToFile(metricsNaive, "metrics/NaiveBayes");
			System.out.println("MSE = " + ClassificationMetrics.computeMSE(trainingDataNaive, modelNaive));

			modelNaive.save(sc.sc(), "models/NaiveBayes");


			NaiveBayesModel sameModelNaive = NaiveBayesModel.load(sc.sc(), "models/NaiveBayes");
	
			//newDataNaive ima parametre kao poslednji podatak u nasem datasetu i on sluzi za testiranje naseg klasifikatora
			//trebalo bi da daje predikciju Yes, odnosno 0.0
			Vector newDataNaive = Vectors.dense(new double[] { 400, 333 ,117, 4, 5, 4, 9.66, 1});
			double predictionNaive = sameModelNaive.predict(newDataNaive);
			System.out.println("Model Prediction for Naive Bayes on New Data = " + predictionNaive);

			
			//KRAJ NAIVE BAYES
			
			//DECISION TREE
			JavaRDD<LabeledPoint>[] splitsDT = parsedData.randomSplit(new double[] { 0.77, 0.23 }, 11L);
			JavaRDD<LabeledPoint> trainingDataDT = splitsDT[0].cache();
			JavaRDD<LabeledPoint> testDataDT = splitsDT[1];

			//metrike koje se delimicno poklapaju sa J48 u weki
			int numClassesDT = 3;
			Map<Integer, Integer> categoricalFeaturesInfoDT = new HashMap<>();
			String impurityDT = "gini";
			int maxDepthDT = 5;
			int maxBinsDT = 32;

			DecisionTreeModel modelDT = DecisionTree.trainClassifier(trainingDataDT, numClassesDT, categoricalFeaturesInfoDT,
					impurityDT, maxDepthDT, maxBinsDT);

			JavaPairRDD<Object, Object> predictionAndLabelsDT = testDataDT
					.mapToPair(p -> new Tuple2<>(modelDT.predict(p.features()), p.label()));
			MulticlassMetrics metricsDT = new MulticlassMetrics(predictionAndLabelsDT.rdd());

			ClassificationMetrics.saveMetricsToFile(metricsDT, "metrics/DecisionTree");

			modelDT.save(sc.sc(), "models/DecisionTree");

			DecisionTreeModel sameModelDT = DecisionTreeModel.load(sc.sc(), "models/DecisionTree");
			
			//newDataDT ima parametre kao poslednji podatak u nasem datasetu i on sluzi za testiranje naseg klasifikatora
			//trebalo bi da daje predikciju Yes, odnosno 0.0
			Vector newDataDT = Vectors.dense(new double[] { 400, 333 ,117, 4, 5, 4, 9.66, 1});

			double predictionDT = sameModelDT.predict(newDataDT);
			
			System.out.println("Model Prediction for Decision Tree on New Data = " + predictionDT);

			sc.close();

			
			
			
		} catch (IOException e) {
			System.out.println("Exception error " + e);
		}

	}


}
