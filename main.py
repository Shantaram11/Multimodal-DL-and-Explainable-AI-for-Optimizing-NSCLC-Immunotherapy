from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.stats import boxcox
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def data_clean_visualization():
    for title in ["pathomics", "radiomics", "transcriptomics"]:
        data_clinical = pd.read_csv("clinical_modified.csv").dropna()
        data_clinical.dropna()
        data_set = pd.read_csv(f"{title}.csv")
        data_merged = data_set.merge(data_clinical, how="left", on="patient_id").dropna(subset=["Best response"])
        label = data_merged["Best response"]
        data = data_merged.drop(columns=["Best response"])
        data["Best response"] = label
        data.to_csv(f"{title}_merged.csv", index=False)

    plt.subplot(2,3, 2)
    data_clinical = pd.read_csv("clinical_targets.csv").dropna()
    mapping_bs = {"Progression": 0, "Stable": 0, "Partial":1, "Complete": 1}
    data_clinical["Best response"] = data_clinical["Best response"].map(mapping_bs)
    print(data_clinical["Best response"].drop_duplicates())
    labels_count_clinical = [sum(data_clinical["Best response"] == 0), sum(data_clinical["Best response"] == 1)]
    plt.pie(labels_count_clinical, labels=["PR/CR", "SD/PD"], autopct="%1.1f%%")
    plt.title("Distribution of clinical data")

    plt.subplot(2,3,4)
    data_pathomics = pd.read_csv("pathomics_merged.csv").dropna()
    labels_count_pathomics = [sum(data_pathomics["Best response"] == 0), sum(data_pathomics["Best response"] == 1)]
    plt.pie(labels_count_pathomics, labels=["PR/CR", "SD/PD"], autopct="%1.1f%%")
    plt.title("Distribution of pathomics data")

    plt.subplot(2,3,5)
    data_radiomics = pd.read_csv("radiomics_merged.csv").dropna()
    labels_count_radiomics = [sum(data_radiomics["Best response"] == 0), sum(data_radiomics["Best response"] == 1)]
    plt.pie(labels_count_radiomics, labels=["PR/CR", "SD/PD"], autopct="%1.1f%%")
    plt.title("Distribution of radiomics data")

    plt.subplot(2,3,6)
    data_transcriptomics = pd.read_csv("transcriptomics_merged.csv").dropna()
    labels_count_transcriptomics = [sum(data_transcriptomics["Best response"] == 0), sum(data_transcriptomics["Best response"] == 1)]
    plt.pie(labels_count_transcriptomics, labels=["PR/CR", "SD/PD"], autopct="%1.1f%%")
    plt.title("Distribution of transcriptomics data")
    plt.show()

def file_merge():
    for title in ["pathomics", "radiomics", "transcriptomics"]:
        data = pd.read_csv(f"{title}_merged.csv")
        print(len(data.columns))
        label = data["Best response"]
        data = data.drop(columns=["patient_id", "Best response"])
        if title == "transcriptomics":
            data = data.drop(columns=["Biopsy site"])
        data = data.fillna(data.median())
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data))
        data = data.fillna(data.mean())
        pca = PCA(n_components=0.90)
        data_dealt = pd.DataFrame(pca.fit_transform(data))
        data_dealt["Best response"] = label
        data_dealt.to_csv(f"{title}_pca.csv", index=False)
        print(f"{title} data with {pca.n_components_}")

def get_labels():
    data_clinical = pd.read_csv("clinical_targets.csv").dropna()
    data_clinical.dropna()
    mapping_bs = {"Progression": 0, "Stable": 0, "Partial":1, "Complete": 1}
    data_clinical["Best response"] = data_clinical["Best response"].map(mapping_bs)
    die_within_1_year = []
    progression_within_6_month = []
    for i in range(len(data_clinical)):
        d = data_clinical.iloc[i]
        vs = d["Vital status"]
        os = d["OS"]
        pro = d["Progression"]
        pfs = d["PFS"]
        if vs == "dead" and os < 365:
            die_within_1_year.append(1)
        else:
            die_within_1_year.append(0)
        if pro == "Yes" and pfs < 182:
            progression_within_6_month.append(1)
        else:
            progression_within_6_month.append(0)

#


def preprocess_data_final(df, ordinal_features=None):
    patient_ids = df["patient_id"] if "patient_id" in df.columns else None
    best_response = df["Best response"] if "Best response" in df.columns else None
    progression_occurrence = df["Progression occurrence"] if "Progression occurrence" in df.columns else None
    potential_status = df["Potential status"] if "Potential status" in df.columns else None
    chemotherapy = df["chemotherapy"] if "chemotherapy" in df.columns else None

    df = df.drop(columns=["patient_id", "Best response", "Progression occurrence", "Potential status", "chemotherapy"],
                 errors="ignore")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    ordinal_cols = ordinal_features if ordinal_features else []

    if numerical_cols:
        imputer_num = SimpleImputer(strategy="mean")
        df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    if categorical_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols].astype(str))

        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)


    for col in numerical_cols:
        if col not in ordinal_cols and df[col].min() > 0:
            df[col], _ = boxcox(df[col] + 1e-6)

    scale_cols = [col for col in numerical_cols if col not in ordinal_cols]
    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    df = df.drop(columns=to_drop)

    if patient_ids is not None:
        df.insert(0, "patient_id", patient_ids)
    if chemotherapy is not None:
        df["chemotherapy"] = chemotherapy
    if best_response is not None:
        df["Best response"] = best_response
    if progression_occurrence is not None:
        df["Progression occurrence"] = progression_occurrence
    if potential_status is not None:
        df["Potential status"] = potential_status

    return df

def final_merge(p, r, t, title):
    data_pathomics = pd.read_csv(p)
    data_radiomics = pd.read_csv(r)
    data_radiomics = data_radiomics.drop(columns = ["Best response", "Progression occurrence", "Potential status", "chemotherapy"])
    data_transcriptomics = pd.read_csv(t)
    data_transcriptomics = data_transcriptomics.drop(columns = ["Best response", "Progression occurrence", "Potential status", "chemotherapy"])
    data_merged = pd.merge(data_transcriptomics, data_radiomics, how="left", on="patient_id").dropna()
    data_final = pd.merge(data_merged, data_pathomics, how="left", on="patient_id").dropna()
    data_final.to_csv(title, index=False)

def get_rfe():
    data = pd.read_csv("pathomics_preprocessed.csv")
    chemotherapy = data["chemotherapy"]
    best_response = data["Best response"]
    progression_occurrence = data["Progression occurrence"]
    potential_status = data["Potential status"]
    labels = data[["Best response", "Potential status", "Progression occurrence"]]
    patient_ids = data["patient_id"]
    data = data.drop(columns=["patient_id", "Best response", "Potential status", "Progression occurrence", "chemotherapy"])
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(rf, n_features_to_select=49)
    rfe.fit(data, labels)
    selected_features = data.columns[rfe.support_]
    data = data[selected_features]
    if patient_ids is not None:
            data.insert(0, "patient_id", patient_ids)
    if chemotherapy is not None:
        data["chemotherapy"] = chemotherapy
    if best_response is not None:
        data["Best response"] = best_response
    if progression_occurrence is not None:
        data["Progression occurrence"] = progression_occurrence
    if potential_status is not None:
        data["Potential status"] = potential_status

def visualiztion_processed_data():
    for file in ["pathomics", "radiomics", "transcriptomics"]:
        file_name = file + "_preprocessed.csv"
        data = pd.read_csv(file_name)
        po = [data["Progression occurrence"].sum(), len(data) - data["Progression occurrence"].sum()]
        ps = [data["Potential status"].sum(), len(data) - data["Potential status"].sum()]
        fig, axes = plt.subplots(1,2)
        axes[0].pie(po, labels=["Otherwise", "Progression occurrence within 6 months"])
        axes[1].pie(ps, labels=["Otherwise", "Die within 1 year"])
        axes[0].set_title("Progression Occurrence")
        axes[1].set_title("Potential Status")
        fig.suptitle(f"Class Distribution for {file}")
        plt.show()

def get_feature_matrix_pca():
    pathomics = pd.read_csv("pathomics_preprocessed.csv")
    patient_ids = pathomics["patient_id"]
    best_response = pathomics["Best response"]
    chemotherapy = pathomics["chemotherapy"]
    progression_occurrence = pathomics["Progression occurrence"]
    potential_status = pathomics["Potential status"]
    pathomics_data = pathomics.drop(columns=["patient_id", "chemotherapy", "Best response", "Progression occurrence", "Potential status"])
    pca = PCA(n_components=50)
    pathomics_pca = pd.DataFrame(pca.fit_transform(pathomics_data))
    feature_matrix = []
    for i in range(50):
        l = []
        components_df = pd.DataFrame(pca.components_, columns=pathomics_data.columns,index=[f'PC{i + 1}' for i in range(pca.n_components_)])
        for j in range(len(components_df.iloc[i])):
            l.append((components_df.columns[j], components_df.iloc[i][j]))
        l_ordered = sorted(l, reverse=True, key=lambda x:abs(x[1]))
        feature_matrix.append(l_ordered[:5])
    pd.DataFrame(feature_matrix).to_csv("Feature_matrix_PCA.csv", index=False)
    if patient_ids is not None:
        pathomics_pca.insert(0, "patient_id", patient_ids)
    if chemotherapy is not None:
        pathomics_pca["chemotherapy"] = chemotherapy
    if best_response is not None:
        pathomics_pca["Best response"] = best_response
    if progression_occurrence is not None:
        pathomics_pca["Progression occurrence"] = progression_occurrence
    if potential_status is not None:
        pathomics_pca["Potential status"] = potential_status
    pathomics_pca.to_csv("pathomics_pca.csv", index=False)

def get_SMOTE(file, target):
    df = pd.read_csv(file)
    length = len(df)
    patient_ids = df["patient_id"]
    best_response = df["Best response"]
    progression_occurrence = df["Progression occurrence"]
    potential_status = df["Potential status"]
    df = df.drop(columns=["patient_id", "Best response", "Progression occurrence", "Potential status"])
    if target == "Best response":
        label = best_response
    elif target == "Progression occurrence":
        label = progression_occurrence
    elif target == "Potential status":
        label = potential_status
    smote = SMOTE(sampling_strategy={0:250, 1:250}, random_state=42)
    X, y = smote.fit_resample(df, label)
    X[target] = y
    data_synthetic = X.iloc[length:]
    title = file[:-4] + "_smote" + "_" + target + ".csv"
    data_synthetic.to_csv(title, index=False)



if __name__ == "__main__":
    for file in ["final_merged_pca.csv", "final_merged_rfe.csv"]:
        for target in ["Best response", "Progression occurrence", "Potential status"]:
            get_SMOTE(file, target)
