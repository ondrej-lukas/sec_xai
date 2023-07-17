import pandas as pd

def load_data_from_csv(filename:str):
    df = pd.read_csv(filename)
    return df