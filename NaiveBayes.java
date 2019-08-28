import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayes {

	public static void main(String[] args) {
		if (args.length != 6) {
			System.out.println(
					"Incorrect number of arguments. Pass arguments in this order: vocabulary.txt, map.csv, train_label.csv, train_data.csv, test_label.csv, test_data.csv.");
			System.exit(0);
		}

		String vocabulary = args[0];
		String map = args[1];
		String trainLabel = args[2];
		String trainData = args[3];
		String testLabel = args[4];
		String testData = args[5];

		DocumentClassifier classifier;

		try {
			classifier = new DocumentClassifier(vocabulary, map, trainLabel, trainData, testLabel, testData);
			classifier.priors();
			classifier.probabilityOfMLE();
			classifier.probabilityOfBE();
			HashMap<Integer, ArrayList<Integer>> trainBE = classifier.estimatedClass(classifier.train_data,
					classifier.probabilityBE);
			HashMap<Integer, ArrayList<Integer>> testBE = classifier.estimatedClass(classifier.test_data,
					classifier.probabilityBE);
			HashMap<Integer, ArrayList<Integer>> testMLE = classifier.estimatedClass(classifier.test_data,
					classifier.probailityMLE);
			classifier.performanceOfTrainingBE(classifier.train_label, trainBE, "trainBE");
			classifier.performanceOFTestBE(classifier.test_label, testBE, "testBE");
			classifier.performanceOfTestSetMLE(classifier.test_label, testMLE, "testMLE");
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
