# Function classifier

In the "Classifier.py" file there is the classifier coded following the graphs feature approach for preprocessing.
In the "Classifier2.py" file , with which I obtained best results the classifier has been coded preprocessing the dataset using the vectorizer.

## Introduction

The aim of this project was to solve a classification problem using a machine learning approach. Each instance in the given dataset is a function that can belong to one of the identified classes: encryption, math, sort, string. In particular each function in the dataset is represented through a dictionary in which, under the key “lista_asm”, there is the list of the assembly instructions of that function.
To solve this problem, the main technologies I have used are:
- Various modules belonging to the scikit learn library
- The networkx library for handling graphs features
- The pandas library for reading the json file in the python code
According to what was told us during the seminar I understood that each function could be represented as control flow graph from which extract some features useful to solve the classification problem.
In addition to this another important aspect, useful for solving the problem, was the fact that each class is distinguished by the others for the spread presence of a certain kind of instruction, for example, the encryption class is characterized by a large diffusion of the XOR instruction in the functions belonging to it.
Thus, when I had to decide which approach to use for preprocessing the dataset, I wanted to try both the possibilities, for comparing them.
For what concern, instead, the machine learning model, also here I wanted to compare different solutions; thus, for each approach to the preprocessing, I’ve tested two different models:
- Support Vector Machine with a linear kernel
- Random Forest

## First approach to preprocessing

In my first trial I approached the preprocess of the dataset trying to extract from it some features from the graphs representing each function. Aware of the fact that I would have to extract these features for each row of the dataset, I decided to read them from the file one per time, using a for loop which iterates over all the file lines, and obtaining from each item read its graph representation
Obtained the graph representation of each assembly code, I started extracting features and, using the networkx library, I have been able to compute the followings:
- Number of nodes
- Number of edges
- Number of assembly instructions
- Diameter of the graph
With the obtained values I have created a list of numerical values, representing each instance, and then storing all this lists in a “bigger” vector, which I gave as the input of the train_test_split function with also a vector containing all the labels of the instances in the dataset.
For splitting the dataset I decided, following what I’ve learned during the course, to assign about one third of the dataset to the test set and the remaining to the training one. In particular I assigned the 30% to the test and the 70% to the training.
As I said before, the two machine learning models I’ve taken in account have been SVM (linear kernel) and Random Forest.

## SVM
The results of the prediction made by the SVM model are:
Accuracy: 0.756 (for this execution)

### Comments
From the confusion matrix is easy to derive the following data:
- Precision average = 0.765
- Recall average = 0.696
- F1-score average = 0.718

Thus, this machine learning model has a non exciting capacity of avoiding to predict samples of one class as other classes and viceversa as evidenced by its low precision and recall medium values. In particular: the prediction of encryption samples is quite good, in fact it has a high f1-score; more problematic are instead the other ones, with particular
attention to the sort class prediction. In fact, from the confusion matrix it is possible to see how the system seems to be not able to distinguish well sort samples from math and string ones (math in particular).
Looking at the f1-score average, which should be a good indicator of the system validity, it is possible to say that this system should be drastically improved.
As a last evaluation of the system I also run the k-cross validation on five different executions.
Accuracy values obtained in the five executions: [0.75631175 0.77332602 0.75631175 0.75246981 0.76344676].

From which the average and standard deviation: Accuracy: 0.760 (+/- 0.01)

## Random Forest
The resuls of the prediction made by the Random Forest model are: Accuracy: 0.908

### Comments
From the Confusion matrix:
- Precision average = 0.900
- Recall average = 0.870
- F1-score average = 0.883

I chosed to use Random Forest for my second trial because it should be resistent to outliers and because I thought that its procedure for deriving patterns from the dataset could be suitable for this problem.Analyzing the results obtained: it is evident that with this model the system has experienced a drastic improvement with respect to the previois one. High values for the averages of precision and recall allows to say that the system has a good capabilty of predicting samples as they real class.
The worst result is still obtained in sort classification, and in particular in its recall. In fact even if the number of sort samples predicted as other classes is about halved with respect to the linear SVM, there is still a 30% of sort samples wrongly predicted.

From the K-Cross Validation: [0.91108672 0.9143798 0.91712404 0.91986828 0.91492865].
Accuracy: 0.915 (+/- 0.01)

## Considerations
During the trials made for all the models presented so far, I tried to consider different subsets of the features I was able to extract and I noticed the importance of the feature which counts the number of the rows of each function.
With respect to the other feature, in fact, if I tried to not take in account this one the results obtained in terms of accuracy decreased at least of the 10%.
Moreover, without this feature, with the svm linear kernel model there were no samples predicted as sort class and also the prediction of encryption samples has been drastically worsened. In fact, using the row count feature, the prediction of encryption samples has a medium f1-score, between the two models, of 0.924, arriving at 0.561 without considering this feature.
Furthermore, doing these trials with features subsets, I also noticed how much Random Forest was less influenced by the presence of a single feature. Trying to not consider one of the features, SVM with linear kernel experienced a drastic worsening of performance, whichever feature I decided to exclude. This did not happen with Random Forest for the most of the trials I made.

## Second approach to preprocessing
Even if considering all the trials made with the previous pre-processing approach the results obtained were not so bad, I decided to try another approach, but still using the same two ML methods, to make a comparison. For this approach I have read the content of the json file using the pandas library.
From what I was able to understand from the seminar, one of the more relevant aspects for recognizing and classifying well the instances, is the number of instructions of a certain type in the assembly code of each instance.
Since I founded a bit tricky to analyze each instance and summing “by hand” all the occurrences of a certain type of instruction contained in the assembly code, I decided to use the vectorizers offered by the scikitlearn library.
In general, the results obtained with this pre processing approach, regardless of the model used, have been really impressing.
For the first trial with this pre processing approach I tried to use the CountVectorizer and again a linear kernel support vector machine model. Even this time I have split the dataset assigning 30% of samples to the test set.

## SVM
The results of the prediction made by the SVM model are: Accuracy: 0.979

### Comments
From the Confusion matrix:
- Precision average = 0.967
- Recall average = 0.964
- F1-score average = 0.966
 
From the K-Cross Validation [0.97969265 0.98133919 0.98737651 0.98024149 0.98518112]. 
Accuracy: 0.983 (+/- 0.01) 

## Random Forest
The resuls of the prediction made by the Random Forest model are:
Accuracy: 0.985

### Comments
From the Confusion Matrix:
- Precision average = 0.985
- Recall average = 0.973
- F1-score average = 0.979

From the K-Cross Validation: [0.98298573 0.98792536 0.9884742 0.98682766 0.9895719].
Accuracy: 0.987 (+/- 0.0045)

## Considerations 
During my trials with this approach to the preprocessing I didn’t have the same freedom in considering different subsets of features as before, in fact the output of the vectorizer was the only feature I was considering. For this reason, the only thing I could do to make some comparisons has been to try different vectorizers, for instance, the TfidfVectorizer and the HashingVectorizer, in addition to the already mentioned CountVectorizer. For what concerns the TfidfVectorizer, the results obtained are very similar to the ones obtained with the CountVectorizer. Talking about the Hashing one, instead, even if in general the results were comparable to the ones obtained with the two other vectorizers, the system, with both the models tried, experienced a quite relevant drop in the results of the sort class classification. In particular, its recall with the SVM model has assumed a value of 0.754. The impressive results obtained using this preprocessing approach convinced me to try using other machine learning models for testing its robustness. For instance, I’ve tried Multinomial Nayve Bayes and DecisionTreeClassifier. I’ m not reporting their results here because they are quite similar to the ones obtained with the aforementioned two models (SVM and Random Forest). Therefor I think I can state that this approach gives a good robustness to the system, guaranteeing very high performances with different models.

## Conclusions 
Analyzing the results obtained in both the approaches to the pre-processing with the two models it is possible to conclude that: - The feature which was fundamental for a correct classification of the functions was the vectorization of the input through the vectorizers, to look for patterns in recurring instructions in the functions. - In both the cases the model which gave the best results has been RandomForest even if in the second approach to preprocessing the two models gave comparable results.
Considering that the best results have been obtained with the classifier that uses Random Forest after having preprocessed the dataset through the use of the CountVectorizer, I run the blind test on this classifier.
