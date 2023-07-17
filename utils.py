import pandas as pd

def load_data_from_csv(filename:str):
    df = pd.read_csv(filename)
    return df

def prepare_data_classification(filename:str)->tuple:
    data = load_data_from_csv("combined_all_days_and_clients.csv")
    x = data.copy()
    x = x.drop(columns=["label", "detailedlabel", "day", "hour", "id.orig_h", "id.resp_h", "id.resp_p", "proto"])
    y = data["label"]
    return x,y