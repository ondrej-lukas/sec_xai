from sklearn.model_selection import train_test_split


def robustness(X, y):
    data = lead_data_from_csv("combined_all_days_and_clients.csv")
    x = data.copy()
    x = x.drop(columns=["label", "detailedlabel", "day", "hour", "id.orig_h", "id.resp_h", "id.resp_p", "proto"])
    y = data["label"]
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    pass