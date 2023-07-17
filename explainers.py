import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import utils
import shap
import sklearn
import models
from sklearn.ensemble import RandomForestRegressor
from lime.lime_text import LimeTextExplainer

def explain_sample_lime(model, datapoint, label, X_train, y_train):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode="classification")

def explain_sample_shap(model,X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return shap_values.values

if __name__ == "__main__":
    data = utils.load_data_from_csv("combined_all_days_and_clients.csv")
    data = data.replace("Benign", 0)
    data = data.replace("Background", 0)
    data = data.replace("Malicious", 1)
    x = data.copy()
    x = x.drop(columns=["label", "detailedlabel", "day", "hour", "id.orig_h", "id.resp_h", "id.resp_p", "proto"])
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.01)
    
    # Prepares a default instance of the random forest regressor
    model = RandomForestRegressor()
    # Fits the model on the data
    model.fit(X_train, y_train)
    #values = explain_sample_shap(rf._model.predict ,X_train, X_test)
    # Fits the explainer
    #explainer = shap.Explainer(model.predict, X_train)
    # Calculates the SHAP values - It takes some time

    #shap_values = explain_sample_shap(model.predict, X_train, X_test)
    

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=list(X_test.columns), class_names=["Benign", "Malicious"], discretize_continuous=True)
    #exp = explainer.explain_instance(X_test[1], model.predict, y_test[1], num_features=5)
    # idx = 1
    # print(X_test[idx])
    # exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=6)