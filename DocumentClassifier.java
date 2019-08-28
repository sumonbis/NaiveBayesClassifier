import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class DocumentClassifier {

	double[] prior;
	int noOfWords = 0, noOfClasses = 0;
	double[][] probailityMLE, probabilityBE;
	HashMap<Integer, Document> train_data = new HashMap<Integer, Document>();
	HashMap<Integer, Document> test_data = new HashMap<Integer, Document>();
	HashMap<Integer, ArrayList<Integer>> train_label = new HashMap<Integer, ArrayList<Integer>>();
	HashMap<Integer, ArrayList<Integer>> test_label = new HashMap<Integer, ArrayList<Integer>>();

	public DocumentClassifier(String vocabulary, String map, String trainLabel, String trainData, String testLabel,
			String testData) {
		int omega = 0, docId = 1, docIndex = 0, wordIndex = 0, count = 0;
		BufferedReader reader = null;

		// read all files
		try {
			reader = new BufferedReader(new FileReader(map));
			while (reader.readLine() != null)
				noOfClasses++;
			reader.close();

			reader = new BufferedReader(new FileReader(vocabulary));
			while (reader.readLine() != null)
				noOfWords++;
			reader.close();

			reader = new BufferedReader(new FileReader(trainData));
			String line, marker = ",";
			while ((line = reader.readLine()) != null) {
				String[] doc = line.split(marker);
				docIndex = Integer.parseInt(doc[0]);
				wordIndex = Integer.parseInt(doc[1]);
				count = Integer.parseInt(doc[2]);
				Document docu = train_data.get(docIndex);
				if (docu == null) {
					docu = new Document(docIndex);
					train_data.put(docIndex, docu);
				}
				Word wrd = new Word(wordIndex, count);
				docu.addWords(wrd, count);
			}
			reader.close();

			reader = new BufferedReader(new FileReader(testData));
			while ((line = reader.readLine()) != null) {
				String[] docData = line.split(marker);
				docIndex = Integer.parseInt(docData[0]);
				wordIndex = Integer.parseInt(docData[1]);
				count = Integer.parseInt(docData[2]);
				Document docu = test_data.get(docIndex);
				if (docu == null) {
					docu = new Document(docIndex);
					test_data.put(docIndex, docu);
				}
				Word word = new Word(wordIndex, count);
				docu.addWords(word, count);
			}
			reader.close();

			reader = new BufferedReader(new FileReader(trainLabel));
			while ((line = reader.readLine()) != null) {
				omega = Integer.parseInt(line);
				if (!train_label.containsKey(omega))
					train_label.put(omega, new ArrayList<Integer>());
				train_label.get(omega).add(docId++);
			}
			reader.close();

			reader = new BufferedReader(new FileReader(testLabel));
			docId = 1;
			while ((line = reader.readLine()) != null) {
				omega = Integer.parseInt(line);
				if (!test_label.containsKey(omega))
					test_label.put(omega, new ArrayList<Integer>());
				test_label.get(omega).add(docId++);

			}
			reader.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	///////////////// File read completed.

	// Calculate priors
	public void priors() {
		prior = new double[noOfClasses];
		int noOfDocs = train_data.size();
		System.out.println("\nClass Priors:\n");
		for (int i = 0; i < noOfClasses; i++) {
			prior[i] = train_label.get(i + 1).size() / (double) noOfDocs;
			System.out.printf("P(Omega = %3d) = %.4f\n", i + 1, prior[i]);
		}
	}

	public void probabilityOfMLE() {
		double wordCount = 0.0;
		int tokens[] = new int[noOfWords];
		probailityMLE = new double[noOfClasses][noOfWords];

		for (int i = 0; i < noOfClasses; i++) {
			List<Integer> documnets = train_label.get(i + 1);
			for (int j = 0; j < documnets.size(); j++) {
				Document doc = train_data.get(documnets.get(j));
				wordCount += doc.getTotalWords();
				List<Word> words_List = doc.getTokens();
				for (int k = 0; k < words_List.size(); k++) {
					Word word = words_List.get(k);
					tokens[word.getWordIndex() - 1] += word.getCount();
				}
			}
			for (int j = 0; j < noOfWords; j++) {
				probailityMLE[i][j] = tokens[j] / wordCount;
			}
		}
	}

	public void probabilityOfBE() {
		int i = 0, j = 0;
		probabilityBE = new double[noOfClasses][noOfWords];
		for (i = 0; i < noOfClasses; i++) {
			double totalWordCount = 0.0;
			List<Integer> documnet_List = train_label.get(i + 1);
			int tokenCount[] = new int[noOfWords];
			for (j = 0; j < documnet_List.size(); j++) {
				Document doc = train_data.get(documnet_List.get(j));
				totalWordCount += doc.getTotalWords();
				List<Word> words_List = doc.getTokens();
				for (int k = 0; k < words_List.size(); k++) {
					Word word = words_List.get(k);
					tokenCount[word.getWordIndex() - 1] += word.getCount();
				}
			}
			for (j = 0; j < noOfWords; j++) {
				probabilityBE[i][j] = (tokenCount[j] + 1) / (totalWordCount + noOfWords);
			}
		}
	}

	public void performanceOfTrainingBE(HashMap<Integer, ArrayList<Integer>> correctLabel,
			HashMap<Integer, ArrayList<Integer>> estimateLabel, String dataType) {
		getPerformanceData(correctLabel, estimateLabel, dataType);
	}

	public void performanceOFTestBE(HashMap<Integer, ArrayList<Integer>> correctLabel,
			HashMap<Integer, ArrayList<Integer>> estimateLabel, String dataType) {
		getPerformanceData(correctLabel, estimateLabel, dataType);
	}

	public HashMap<Integer, ArrayList<Integer>> estimatedClass(HashMap<Integer, Document> doc,
			double[][] predictClass) {
		int i = 0;
		int classID = 0;
		HashMap<Integer, ArrayList<Integer>> estimateLabel = new HashMap<Integer, ArrayList<Integer>>();
		List<Double> predictions = new ArrayList<Double>();
		double value = 0.0, max = 0.0;
		for (HashMap.Entry<Integer, Document> entry : doc.entrySet()) {
			Document docu = entry.getValue();
			for (i = 0; i < noOfClasses; i++) {
				List<Word> tokenCount = docu.getTokens();
				value += Math.log(prior[i]);
				for (Word token : tokenCount) {
					double likelihood = Math.log(predictClass[i][token.getWordIndex() - 1]);
					value += (token.getCount() * likelihood);
				}
				predictions.add(value);
				value = 0.0;
			}
			max = Collections.max(predictions);
			classID = predictions.indexOf(max) + 1;
			if (!estimateLabel.containsKey(classID)) {
				estimateLabel.put(classID, new ArrayList<Integer>());
			}
			estimateLabel.get(classID).add(docu.getDocumentIndex());
			predictions = new ArrayList<Double>();
		}
		return estimateLabel;
	}

	public void performanceOfTestSetMLE(HashMap<Integer, ArrayList<Integer>> correctClass,
			HashMap<Integer, ArrayList<Integer>> estimateLabel, String type) {
		getPerformanceData(correctClass, estimateLabel, type);
	}

	public void getPerformanceData(HashMap<Integer, ArrayList<Integer>> correctLabel,
			HashMap<Integer, ArrayList<Integer>> estimatedLabel, String type) {
		double correctDocs = 0.0;
		double docs = 0.0;
		double[] classAccuracy = new double[noOfClasses];
		String[] traingCriteria = { "trainBE", "testBE", "testMLE" };
		for (int i = 1; i <= noOfClasses; i++) {
			ArrayList<Integer> actualDocuments = correctLabel.get(i);
			ArrayList<Integer> estimateDocuments = estimatedLabel.get(i);
			ArrayList<Integer> correctDocuments = new ArrayList<>(estimateDocuments);
			correctDocuments.retainAll(actualDocuments);
			correctDocs += correctDocuments.size();
			docs += actualDocuments.size();
			classAccuracy[i - 1] = (double) correctDocuments.size() / actualDocuments.size();
		}

		if (type.equals(traingCriteria[0])) {
			printTrainingOverall(correctDocs, docs);
			printTrainingClass(classAccuracy);
			printTrainingConfusionMatrix(correctLabel, estimatedLabel);
		} else if (type.equals(traingCriteria[1])) {
			printTestOverall(correctDocs, docs);
			printTestClass(classAccuracy);
			printTestConfusionMatrix(correctLabel, estimatedLabel);
		} else if (type.equals(traingCriteria[2])) {
			printTestOverallMLE(correctDocs, docs);
			printTestClassMLE(classAccuracy);
			printTestConfusionMatrixMLE(correctLabel, estimatedLabel);
		}
	}

	///////////////
	////// print methods
	public void printTrainingOverall(double noOfCorrectDocs, double noOfDocs) {
		double overallAccuracy = 0.0;
		overallAccuracy = noOfCorrectDocs / noOfDocs;
		System.out.printf("\n\nOverall Accuracy for Training Dataset = %.4f\n\n", overallAccuracy);
	}

	public void printTrainingClass(double[] classAccuracy) {
		System.out.println("Class Accuracy for Training Dataset:\n");
		for (int i = 0; i < classAccuracy.length; i++)
			System.out.printf("Group %2d = %.4f\n", i + 1, classAccuracy[i]);
		System.out.println("\n");

	}

	public void printTrainingConfusionMatrix(HashMap<Integer, ArrayList<Integer>> correctLabel,
			HashMap<Integer, ArrayList<Integer>> estimateLabel) {
		int matrix[][] = new int[noOfClasses][noOfClasses];
		System.out.println("Confusion Matrix for Training Dataset:\n");
		for (int i = 0; i < matrix.length; i++) {
			List<Integer> actualDocs = correctLabel.get(i + 1);
			for (int j = 0; j < matrix.length; j++) {
				List<Integer> value = new ArrayList<>(estimateLabel.get(j + 1));
				value.retainAll(actualDocs);
				matrix[i][j] = value.size();
				System.out.printf("%5d ", matrix[i][j]);
			}
			System.out.println('\n');
		}
	}

	public void printTestOverall(double totalCorrectDocs, double totalDocs) {
		double overallAccuracy = 0.0;
		overallAccuracy = totalCorrectDocs / totalDocs;
		System.out.printf("\n\nOverall Accuracy for Test Dataset = %.4f\n\n", overallAccuracy);
	}

	public void printTestClass(double[] classAccuracy) {
		System.out.println("Class Accuracy for Test Dataset:\n");
		for (int i = 0; i < classAccuracy.length; i++)
			System.out.printf("Group %2d = %.4f\n", i + 1, classAccuracy[i]);
		System.out.println("\n");

	}

	public void printTestConfusionMatrixMLE(HashMap<Integer, ArrayList<Integer>> correctLabel,
			HashMap<Integer, ArrayList<Integer>> estimateLabel) {
		int matrix[][] = new int[noOfClasses][noOfClasses];
		System.out.println("Confusion Matrix for Test Dataset using MLE:\n");
		for (int i = 0; i < matrix.length; i++) {
			List<Integer> actualDocs = correctLabel.get(i + 1);
			for (int j = 0; j < matrix.length; j++) {
				List<Integer> value = new ArrayList<>(estimateLabel.get(j + 1));
				value.retainAll(actualDocs);
				matrix[i][j] = value.size();
				System.out.printf("%5d ", matrix[i][j]);
			}
			System.out.println('\n');
		}
	}

	public void printTestConfusionMatrix(HashMap<Integer, ArrayList<Integer>> correctLabel,
			HashMap<Integer, ArrayList<Integer>> estimateLabel) {
		int matrix[][] = new int[noOfClasses][noOfClasses];
		System.out.println("Confusion Matrix for Test Dataset:\n");
		for (int i = 0; i < matrix.length; i++) {
			List<Integer> actualDocs = correctLabel.get(i + 1);
			for (int j = 0; j < matrix.length; j++) {
				List<Integer> value = new ArrayList<>(estimateLabel.get(j + 1));
				value.retainAll(actualDocs);
				matrix[i][j] = value.size();
				System.out.printf("%5d ", matrix[i][j]);
			}
			System.out.println('\n');
		}
	}

	public void printTestOverallMLE(double totalCorrectDocs, double totalDocs) {
		double overallAccuracy = 0.0;
		overallAccuracy = totalCorrectDocs / totalDocs;
		System.out.printf("\n\nOverall Accuracy for Test Dataset using MLE = %.4f\n\n", overallAccuracy);
	}

	public void printTestClassMLE(double[] classAccuracy) {
		System.out.println("Class Accuracy for Test Dataset using MLE:\n");
		for (int i = 0; i < classAccuracy.length; i++)
			System.out.printf("Group %2d = %.4f\n", i + 1, classAccuracy[i]);
		System.out.println("\n");

	}
}

class Word {
	private int wordIndex;
	private int count;

	public Word(int word_Index, int count) {
		this.wordIndex = word_Index;
		this.count = count;
	}

	public int getCount() {
		return count;
	}

	public int getWordIndex() {
		return wordIndex;
	}

	public void setWordIndex(int wordIdx) {
		this.wordIndex = wordIdx;
	}

	public void setCount(int count) {
		this.count = count;
	}
}

class Document {
	private int DocumentIndex;
	int wordCount = 0;
	private List<Word> token;

	public Document(int Document_Index) {
		this.DocumentIndex = Document_Index;
		token = new ArrayList<Word>();
	}

	public void setTotalWords(int totalWords) {
		this.wordCount = totalWords;
	}

	public void addWords(Word word, int count) {

		token.add(word);
		wordCount += count;
	}

	public List<Word> getTokens() {
		return token;
	}

	public int getTotalWords() {
		return wordCount;
	}

	public int getDocumentIndex() {
		return DocumentIndex;
	}

	public void setDocumentIndex(int documentIdx) {
		this.DocumentIndex = documentIdx;
	}
}
