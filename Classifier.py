import json
import networkx as nx
from networkx import json_graph
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
import ast
from sklearn import metrics
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def row_counter(d):
    #Contatore righe
    count = len(ast.literal_eval(d['lista_asm']))
    return count
    
def compute_diameter(graph):
    try:
        diameter = nx.diameter(graph.to_undirected(), e = None, usebounds = False)
        return diameter
    except:
        #print("infinite path length")
        return 100000000
        
def main():
    features = []
    labels = []
    
    #choose the model you want to use
    model = "svm"     #svm ; randomForest
    if(model == "svm"):
        clf = svm.SVC(kernel = 'linear')
    elif(model == "randomForest"):
        clf = RandomForestClassifier()
        
    with open("pathToDataset", 'r') as f:
        lines = f.readlines()
        for item in lines:
            dfeature = []
            
            for i in range(len(dfeature)):
                dfeature.remove(dfeature[i])
                
            #obtaining graph
            d = json.loads(item)
            nx_graph = json_graph.adjacency_graph(d['cfg'])
            
            #graph features
            dfeature.append(nx.number_of_nodes(nx_graph))
            dfeature.append(nx.number_of_edges(nx_graph))
            dfeature.append(compute_diameter(nx_graph))
            dfeature.append(row_counter(d))
            
            labels.append(d["semantic"])
            features.append(dfeature)
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state = 20)
        
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
        scores = cross_val_score(clf, features, labels, cv=cv)
        print(scores)
        print("Accuracy: %0.3f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        
if __name__ == "__main__":
    main()
