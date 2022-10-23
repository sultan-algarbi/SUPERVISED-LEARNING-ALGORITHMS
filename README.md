# SUPERVISED-LEARNING-ALGORITHMS

## INTRODUCTION:
This project is to experiment with real-world datasets to explore how algorithms for machine learning can be used to identify patterns in data.
It is expected to use supervised learning approaches or paradigms, support vector machine, linear discriminant analysis, k-nearest neighbors, and random forest algorithms have been used.
The models have been trained using k-fold cross-validation (including leave-one-out cross-validation). More than one metric has been used to measure the performances of the models, which are:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Support
•	Confusion Matrix

## LIST OF THE DATASETS:
The system uses the following datasets to train and test this project:

### Students Performance
This data set consists of the marks secured by the students in various subjects.
kaggle.com/spscientist/students-performance-in-exams

### Pima Indians Diabetes
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
www.kaggle.com/uciml/pima-indians-diabetes-database

### Bill Authentication
Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.
archive.ics.uci.edu/ml/datasets/banknote+authentication

### Covid19 Tweets
These tweets are collected using Twitter API and a Python script. A query for this high-frequency hashtag (#covid19) is run on a daily basis for a certain time period, to collect a larger number of tweets samples.
www.kaggle.com/gpreda/covid19-tweets

### Hashtag Donald Trump
These tweets are collected using Twitter API and a Python script. A query for this high-frequency hashtag (#donaldtrump) is run on a daily basis for a certain time period, to collect a larger number of tweets samples.
www.kaggle.com/gpreda/hashtag_donaldtrump

## LIST OF THE METHODS AND ALGORITHMS:
### K-Nearest Neighbors Algorithm (KNN):
The algorithm of the k-nearest neighbor is a non-parametric approach used for classification and regression, with the input consisting of the k closest training examples in the feature space in both cases. K-NN is a type of instance-based learning, or lazy learning, in which the function is only locally approximated, and all computation is deferred until the evaluation of the function. As this algorithm relies on distance for classification, it can significantly improve its accuracy by normalizing the training data.

### Linear Discriminant Analysis Algorithm (LDA):
Linear discriminant analysis (LDA) is an algorithm used to find a linear combination of features that characterizes or distinguishes two or more classes of items or events in machine learning and other fields. The resulting combination can be used as a linear classifier or more generally before subsequent classification for dimensionality reduction. LDA functions because continuous quantities are the measurements made for each observation on independent variables.
 
### Support Vector Machine (SVM):
Support vector machines (SVM) in machine learning are supervised learning models with associated learning algorithms that analyze data used for the analysis of classification and regression. It provides one of the most robust methods of prediction. An SVM training algorithm produces a model that assigns new examples to a class or another, making it a non-probabilistic binary linear classifier, given a set of training examples, each marked as belonging to one or the other of two classes. An SVM model is a representation of the examples as points in space, mapped such that a simple distance that is as large as possible separates the examples of the individual classes. In the same space, new examples are then mapped and predicted to belong to a class based on the side of the gap on which they fall.

### Random Decision Forests:
Random decision forests or random forests are an ensemble learning method for classification, regression, and other tasks in machine learning that operate through the creation at training time of a multitude of decision trees and the development of the class that is the class model (classification) or the individual trees' mean/average prediction (regression). Random forests of decision making correct the practice of overfitting their training set for decision trees. In general, random forests outperform decision trees, but their precision is lower than trees improved by gradients. Data features, however, may influence their performance. Forests are like the bringing together of decision tree algorithm efforts. In this way, the teamwork of multiple trees increases the effectiveness of a single random tree. Forests have the results of k-fold cross-validation, but not very similar.

## RESOURCES:

List of main resources that you have used:
•	https://www.w3schools.com/python
•	https://colab.research.google.com/
•	https://stackoverflow.com/
•	https://stackabuse.com/
•	https://github.com/jupyter/notebook/issues/2030
•	https://jupyter-contrib-nbextensions.readthedocs.io/
•	https://www.xspdf.com/resolution/54819026.html
•	https://www.kaggle.com/
•	https://github.com/twintproject/twint
