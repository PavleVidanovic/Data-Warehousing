import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.ClassificationModel;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;



public class ClassificationMetrics {

	
	public static double computeMSE(JavaRDD<LabeledPoint> trainingData, ClassificationModel model) {

		double meanSquaredError = 0;

		JavaPairRDD<Double, Double> valuesAndPreds = trainingData
				.mapToPair(point -> new Tuple2<>(model.predict(point.features()), point.label()));

		meanSquaredError = valuesAndPreds.mapToDouble(pair -> {
			double difference = pair._1() - pair._2();
			return difference * difference;
		}).mean();

		return meanSquaredError;
	}

	public static void saveMetricsToFile(MulticlassMetrics metrics, String path) throws IOException {
		try {

			Map<Double, String> map = new HashMap<Double, String>();
			map.put(0.0, "Yes");
			map.put(1.0, "Depends");
			map.put(2.0, "No");
			BufferedWriter out = new BufferedWriter(new FileWriter(path));
			for (int i = 0; i < metrics.labels().length; i++) {
				out.write("Class " + map.get((double) metrics.labels()[i]) + " precision = "
						+ metrics.precision(metrics.labels()[i]) + "\n");
				out.write("Class " + map.get((double) metrics.labels()[i]) + " recall = "
						+ metrics.recall(metrics.labels()[i]) + "\n");
				out.write("Class " + map.get((double) metrics.labels()[i]) + " F1 score = "
						+ metrics.fMeasure(metrics.labels()[i]) + "\n");
			}

			out.write("\n");

			
			out.write("\nWeighted precision: " + metrics.weightedPrecision());
			out.write("\nWeighted recall: " + metrics.weightedRecall());
			out.write("\nWeighted F1 score: " + metrics.weightedFMeasure());
			out.write("\nWeighted false positive rate: " + metrics.weightedFalsePositiveRate());
			out.write("\nModel Accuracy on Test Data: " + metrics.accuracy());
			Matrix confusion = metrics.confusionMatrix();
			out.write("\n");
			out.write("\nConfusion matrix: \n" + confusion.toString());

			out.close();
			System.out.println("File created successfully");
			
		}

		catch (IOException e) {
			System.out.println("File not created, " + e.getMessage());
		}

	}
	
	
}
