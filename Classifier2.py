from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

vectorizer = CountVectorizer()
dataset = pd.read_json("pathToDataset", lines = True)
X = vectorizer.fit_transform(dataset.lista_asm)

#choose the model you want to use
model = "randomForest"     #svm ; randomForest
if(model == "svm"):
    clf = svm.SVC(kernel = 'linear')
elif(model == "randomForest"):
    clf = RandomForestClassifier()

#training and prediction on test set
X_train, X_test, y_train, y_test = train_test_split(X, dataset.semantic, test_size = 0.3, random_state = 88)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test,y_pred))

classes = ["encryption", "math", "sort", "string"]
disp = plot_confusion_matrix(clf, X_test, y_test,
                            display_labels=classes,
                            cmap = plt.cm.Blues,
                            values_format="g")
plt.show()

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=20)
scores = cross_val_score(clf, X, dataset.semantic, cv=cv)
print(scores)
print("Accuracy: %0.3f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
#save of the trained model
filename = 'trainedClassifier.sav'
pickle.dump(clf, open(filename, 'wb'))

#blind test
execBlind = input('Do you want to execute the blind test? y/n')
if (execBlind == 'y' or execBlind == 'Y'):
    blindtest = pd.read_json("pathToBlindTest", lines = True)

    blindlist = []
    for row in blindtest.itertuples():
        blindlist.append(row.lista_asm)
        
    blindforpred = vectorizer.transform(blindlist)
    prediction = clf.predict(blindforpred)

    f = open("blindtestOut", "w")
    for pred in prediction:
        f.write(pred)
        f.write("\n")
else:
    print('Thank you!')
