import pandas as pd
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier

def lead_data_from_csv(filename:str):
    df = pd.read_csv(filename)
    return df

class Model:
    
    def __init__(self, name:str) -> None:
        self.name = name
        self._model = None
    
    def save_to_pickle(self) -> None:
        with open(f"{self.name}_model.pickle", "wb") as f:
            pickle.dump(self._model, f)

    def load_from_pickle(self, filename)->None:
        with open(filename, "rb") as f:
            self._model = pickle.load(f)


class RandomForrest(Model):
    def __inti__(self, name="rf", n_estimators:int=10, max_depth=None, min_samples_split=2) -> None:
        super().__init__(name)
        self._model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    
    def fit(self, X,y):
        self._model.fit(X, y)


if __name__ == "__main__":
    data = lead_data_from_csv("combined_all_days_and_clients.csv")
    x = data.copy()
    x = x.drop(columns=["label", "detailedlabel", "day", "hour", "id.orig_h", "id.resp_h", "id.resp_p", "proto"])
    y = data["label"]

    rf = RandomForrest("rf")
    rf.load_from_pickle("rf_model.pickle")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    # clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(rf._model, X_train, y_train, cv=10)
    print(f"RF: {scores.mean()}")
    # with open("rf_model.pickle", "wb") as f:
    #     pickle.dump(clf, f)
    # print(f"RF: {scores.mean()}")
    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # scores = cross_val_score(clf, x, y, cv=10)
    # print(f"SVM: {scores.mean()}")
    # with open("svm_model.pickle", "wb") as f:
    #     pickle.dump(clf, f)
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.9, max_depth=None, random_state=0)
    # with open("GBT_model.pickle", "wb") as f:
    #     pickle.dump(clf, f)
    # scores = cross_val_score(clf, x, y, cv=10)
    # print(f"GBT: {scores.mean()}")