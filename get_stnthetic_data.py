import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def generate_synthetic_data(train_set_ratio, target, k):
    data = pd.read_csv("final_merged_rfe.csv")
    best_response = data["Best response"]
    potential_status = data["Potential status"]
    progression_occurrence = data["Progression occurrence"]
    data = data.drop(columns=["patient_id", "Best response", "Progression occurrence", "Potential status"])
    if target == "bs":
        label = best_response
    elif target == "ps":
        label = potential_status
    elif target == "po":
        label = progression_occurrence
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=train_set_ratio, stratify=label)
    test_set = pd.concat([X_test, y_test], axis=1)
    smote = SMOTE(sampling_strategy={0: 200, 1: 200}, random_state=42, k_neighbors=k)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    train_set = pd.concat([X_train_smote, y_train_smote], axis=1)
    return train_set, test_set


# Call this function to get split data for training and testing set
# parameter train_set_ratio is the ratio for training set. recommend range is [0.3, 0.7] because our original data has imbalance classes
# parameter k is for SMOTE, since SMOTE is a method based on K-nn. recommend range is [2, 5]
# Since we only use partial data to generate synthetic data, I shrank the size of training set from 500 into 400
# Reminder: if get error like "ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 9, n_samples_fit = 7, n_samples = 7", which means k is so large. Use smaller k and try again
def get_synthetic_data(train_set_ratio=0.6, k=2):
    train_set_bs, test_set_bs = generate_synthetic_data(train_set_ratio, "bs", k)
    train_set_po, test_set_po = generate_synthetic_data(train_set_ratio, "po", k)
    train_set_ps, test_set_ps = generate_synthetic_data(train_set_ratio, "ps", k)
    return train_set_bs, test_set_bs, train_set_ps, test_set_ps, train_set_po, test_set_po
