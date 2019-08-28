# NaiveBayesClassifier
Implementation of Naive Bayes classifiers for classifying text documents.

## Dataset
The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. It was originally collected by Ken Lang, probably for his Newsweeder: Learning to filter netnews[1] paper.The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering. The data is organized into 20 different newsgroups, each corresponding to a different topic.

## Implementation
I have implemented Java program that takes the six files as input, builds a Naive Bayes classifier and outputs relevant statistics. I have built the Naive Bayes classifier from the training data (train label.csv, train data.csv), then evaluated its performance on the testing data (test label.csv, test data.csv).

Instructions to run the program:

1. Copy all *.java files to one directory.
2. Place the data files in the same directory
3. Use command line to run.
	i. cd to the directory. 
	ii. Compile : $ javac *.java
	iii. Run: $ java NaiveBayes vocabulary.txt map.csv train_label.csv train_data.csv test_label.csv test_data.csv


## References
1. Ken Lang, Newsweeder: Learning to filter netnews, Proceedings of the Twelfth International Conference on Machine Learning, 331-339 (1995).
