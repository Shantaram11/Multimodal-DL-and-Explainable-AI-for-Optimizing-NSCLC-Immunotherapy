import pandas as pd
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from ctgan import CTGAN



def generate_synthetic_data(n, file, train_set_ratio, target, k):
    data = pd.read_csv(file)
    best_response = data["Best response"]
    potential_status = data["Potential status"]
    progression_occurrence = data["Progression occurrence"]
    try:
        data = data.drop(columns=["patient_id", "Best response", "Progression occurrence", "Potential status"])
    except:
        data = data.drop(columns=["Best response", "Progression occurrence", "Potential status"])
    if target == "bs":
        label = best_response
    elif target == "ps":
        label = potential_status
    elif target == "po":
        label = progression_occurrence
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=train_set_ratio, stratify=label)
    test_set = pd.concat([X_test, y_test], axis=1)
    smote = SMOTE(sampling_strategy={0: n//2, 1: n//2}, random_state=42, k_neighbors=k)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    train_set = pd.concat([X_train_smote, y_train_smote], axis=1)
    return train_set, test_set


# Call this function to get split data in for training and testing set
# parameter train_set_ratio is the ratio for training set. recommend range is [0.3, 0.7] because our original data has imbalance classes
# parameter k is for SMOTE, since SMOTE is a method based on K-nn. recommend range is [2, 5]
# Since we only use partial data to generate synthetic data, I shrank the size of training set from 500 into 400
# Reminder: if get error like "ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 9, n_samples_fit = 7, n_samples = 7", which means k is so large. Use smaller k and try again
def get_synthetic_data(n, file, train_set_ratio=0.6, k=2):
    train_set_bs, test_set_bs = generate_synthetic_data(n, file, train_set_ratio, "bs", k)
    train_set_po, test_set_po = generate_synthetic_data(n, file, train_set_ratio, "po", k)
    train_set_ps, test_set_ps = generate_synthetic_data(n, file, train_set_ratio, "ps", k)
    return train_set_bs, test_set_bs, train_set_ps, test_set_ps, train_set_po, test_set_po

def generate_synthetic_data_CTGAN(n, file, train_set_ratio, target):
    data = pd.read_csv(file)
    if target == "bs":
        label, c = data["Best response"], "Best response"
    elif target == "ps":
        label, c = data["Potential status"], "Potential status"
    elif target == "po":
        label, c = data["Progression occurrence"], "Progression occurrence"
    try:
        data = data.drop(columns=["patient_id", "Best response", "Potential status", "Progression occurrence"])
    except:
        data = data.drop(columns=["Best response", "Potential status", "Progression occurrence"])
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=train_set_ratio, stratify=label)
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    model = CTGAN(epochs=100)
    try:
        model.fit(train_set, discrete_columns=[c, "chemotherapy"])
    except:
        model.fit(train_set, discrete_columns=[c])
    synthetic_data = model.sample(n)
    train_set = pd.concat([train_set, synthetic_data], axis=0)
    return train_set, test_set

def get_synthetic_data_CTGAN(n, file, train_set_ratio=0.6):
    train_set_bs, test_set_bs = generate_synthetic_data_CTGAN(n, file, train_set_ratio, "bs")
    train_set_po, test_set_po = generate_synthetic_data_CTGAN(n, file, train_set_ratio, "po")
    train_set_ps, test_set_ps = generate_synthetic_data_CTGAN(n, file, train_set_ratio, "ps")
    return train_set_bs, test_set_bs, train_set_ps, test_set_ps, train_set_po, test_set_po

def get_description(glossary_path, feature_list):
    glossary = pd.read_csv(glossary_path)
    return list(glossary[glossary["Feature"].isin(feature_list)]["Description"])

if __name__ == "__main__":
    descriptions = get_description("final_glossary.csv", feature_list=["Best response", "Potential status"])
    print(descriptions)