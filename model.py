import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from get_stnthetic_data import *
import pickle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Change the input here
def input(train):
    X = train.iloc[:, :-1].values
    print(X.shape)

    # Standardize the features
    X_scaled = X

    target_col = train.columns[-1]
    print(target_col)
    y = train[target_col].values
    print("Label distribution:", Counter(y))
    return X_scaled, y
class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims=[16, 8, 4], dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def cross_validate_models(X_scaled, y, model_configs, num_epochs=100, batch_size=64):
    results = []

    for config in model_configs:
        print(f"\n🔍 Testing model config: {config}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_f1_scores, val_f1_scores = [], []
        train_acc_scores, val_acc_scores = [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size,
                                      shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

            model = MLP(input_size=X_scaled.shape[1], hidden_dims=config["hidden_dims"], dropout=config["dropout"])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
            # scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

            all_train_preds, all_train_labels = [], []
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # scheduler.step()

            train_f1 = f1_score(all_train_labels, all_train_preds)
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            train_f1_scores.append(train_f1)
            train_acc_scores.append(train_acc)

            # Evaluation
            model.eval()
            all_val_preds, all_val_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            val_f1 = f1_score(all_val_labels, all_val_preds)
            val_acc = accuracy_score(all_val_labels, all_val_preds)
            val_f1_scores.append(val_f1)
            val_acc_scores.append(val_acc)

            print(f"Fold {fold + 1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

        config['avg_train_f1'] = np.mean(train_f1_scores)
        config['avg_val_f1'] = np.mean(val_f1_scores)
        config['avg_train_acc'] = np.mean(train_acc_scores)
        config['avg_val_acc'] = np.mean(val_acc_scores)

        print(
            f"✅ Config {config['hidden_dims']} → Train acc: {config['avg_train_acc']:.4f}, Val acc: {config['avg_val_acc']:.4f}")
        print(
            f"✅ Config {config['hidden_dims']} → Train F1: {config['avg_train_f1']:.4f}, Val F1: {config['avg_val_f1']:.4f}")

        results.append(config)

    return sorted(results, key=lambda x: x['avg_val_f1'], reverse=True)
def train_final_model(X_scaled, y, hidden_dims, dropout=0.3, num_epochs=100, batch_size=64, save_path="final_model.pt"):
    print("\n🚀 Training final model with best config...")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = MLP(input_size=X_scaled.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        #scheduler.step()
    torch.save(model.state_dict(), save_path)
    print(f"✅ Final model saved to {save_path}")
    return model

def evaluate_on_test(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = (outputs > 0.5).float().cpu().numpy()
        labels = y_test_tensor.cpu().numpy()

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f"\n🧪 Final Evaluation on Test Set:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    set_seed(100)
    train_set_bs, test_set_bs, train_set_ps, test_set_ps, train_set_po, test_set_po = get_synthetic_data(n=600,file="final_merged_rfe.csv",train_set_ratio=0.7,k=3)
    X_scaled, y = input(train_set_po)
    model_configs = [
        {"hidden_dims": [16, 8, 4], "dropout": 0.3},
        {"hidden_dims": [32, 16], "dropout": 0.3},
        {"hidden_dims": [64, 32], "dropout": 0.2},
        {"hidden_dims": [16, 8], "dropout": 0.3}
    ]
    sorted_configs = cross_validate_models(X_scaled, y, model_configs)
    best_config = sorted_configs[0]
    print(f"best config: {best_config}")
    final_model = train_final_model(X_scaled, y, hidden_dims=best_config['hidden_dims'], dropout=best_config['dropout'])
    X_test, y_test = input(test_set_po)
    evaluate_on_test(final_model, X_test, y_test)
    # torch.save(final_model, "model_po.pth")

    # X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # data = pd.read_csv("final_merged_rfe.csv").drop(columns=["patient_id", "Best response", "Potential status", "Progression occurrence"])
    # feature_importance = run_shap.run_shap_on_model(final_model, X_train_tensor, X_test_tensor, data.columns, "Potential status",False)
    # feature_string, shap_values =  run_shap.get_top_num_features(5, feature_importance, "Potential status")
    # df = pd.DataFrame({"Features": feature_string, "Shap": shap_values})
    # df.to_csv("shap_po.csv", index=False)
