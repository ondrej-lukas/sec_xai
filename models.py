import numpy as np
import sklearn
import json
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import xgboost as xgb
from xgboost import XGBClassifier
def load_data(filename:str)-> tuple:
    "Loads data from JSON file"
    with open(filename, "r") as f:
        data = json.load(f)
        return data["X"], data["y"], data["4tuples"]

def preprocess_data(x,y)->tuple:
    #convert to numpy arrays
    x = np.array(x,dtype=float)
    f = lambda l: 1 if l in "Malicious" else  0
    y = np.array([f(l) for l in y])
    #normalize 
    x = normalize(x,axis=0)
    return x,y


def linear_model(x,y):
    raise NotImplementedError

def xgboost(X_train, X_test, y_train, y_test):
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"XGB RESULTS: TN={tn/y_test.shape[-1]}, FP={fp/y_test.shape[-1]}, FN={fn/y_test.shape[-1]}, TP={tp/y_test.shape[-1]}, ACC={accuracy_score(y_test, pred)}, Prec={precision_score(y_test, pred)}, Rec={recall_score(y_test, pred)}")
    return clf

def svm(X_train, X_test, y_train, y_test):
    clf = SVC(kernel="linear", class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"SVM RESULTS: TN={tn/y_test.shape[-1]}, FP={fp/y_test.shape[-1]}, FN={fn/y_test.shape[-1]}, TP={tp/y_test.shape[-1]}, ACC={accuracy_score(y_test, pred)}, Prec={precision_score(y_test, pred)}, Rec={recall_score(y_test, pred)}")
    return clf

if __name__ == '__main__':
    x, y, ids = load_data("merged_dataset.json")
    
    x, y = preprocess_data(x,y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=26)
    print(np.sum(y_train==0), np.sum(y_train==1))
    SVM = svm(X_train, X_test, y_train, y_test)
    XGB = xgboost(X_train, X_test, y_train, y_test)
