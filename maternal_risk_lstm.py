import os
import io
import copy
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

# 1. Reproducibility Setup (Seed 42 as reported in Section 5.1.4)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")  # CPU environment reported in Section 4.4.2 & 5.1.4


# 2. Fetch / Load the Mendeley Dataset (DOI: 10.17632/ckhgtgdg4f.1)
def fetch_mendeley_dataset(local_fallback="maternal_health_mendeley.csv"):
    """
    Fetches the Maternal Health Risk dataset (Hossain, 2025) directly via
    the Mendeley Data public API, or loads from local disk if already cached.
    """
    if os.path.exists(local_fallback):
        print(f"Loading cached dataset from {local_fallback}...")
        return pd.read_csv(local_fallback)

    api_url = "https://data.mendeley.com/public-api/datasets/ckhgtgdg4f/files?folder_id=root"
    print("Fetching dataset metadata from Mendeley Data API...")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    files = resp.json()

    # Locate the tabular CSV/Excel file in the dataset repository
    target_file = next(
        (f for f in files if f["filename"].lower().endswith((".csv", ".xlsx", ".xls"))),
        None,
    )
    if not target_file:
        raise FileNotFoundError("No CSV or Excel file found in Mendeley dataset.")

    download_url = target_file["content_details"]["download_url"]
    filename = target_file["filename"]
    print(f"Downloading {filename}...")
    file_resp = requests.get(download_url, timeout=60)
    file_resp.raise_for_status()

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_resp.content))
    else:
        df = pd.read_excel(io.BytesIO(file_resp.content))

    df.to_csv(local_fallback, index=False)
    print(f"Saved local copy to {local_fallback} ({len(df)} rows).")
    return df


# 3. Preprocessing and Sequence Serialization
def prepare_data(df):
    """
    Aligns the 8 clinical features in the exact order specified in Equation (1),
    performs a stratified 70:15:15 split, fits MinMaxScaler on training data only,
    and reshapes static records into (N, 8, 1) sequences.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Map possible column name variations in the Mendeley file to the 8 target features
    col_mapping = {}
    for col in df.columns:
        cl = col.lower().replace("_", "").replace(" ", "")
        if "age" in cl:
            col_mapping[col] = "Age"
        elif "temp" in cl:
            col_mapping[col] = "BodyTemp"
        elif "heart" in cl or "pulse" in cl:
            col_mapping[col] = "HeartRate"
        elif "sys" in cl:
            col_mapping[col] = "SystolicBP"
        elif "dia" in cl:
            col_mapping[col] = "DiastolicBP"
        elif "bmi" in cl:
            col_mapping[col] = "BMI"
        elif "hba1c" in cl:
            col_mapping[col] = "HbA1c"
        elif "fasting" in cl or "glucose" in cl or cl == "bs":
            col_mapping[col] = "FastingGlucose"
        elif "risk" in cl or "label" in cl or "target" in cl or "status" in cl:
            col_mapping[col] = "RiskLevel"

    df = df.rename(columns=col_mapping)

    # Exact feature order defined in Equation (1)
    feature_cols = [
        "Age",
        "BodyTemp",
        "HeartRate",
        "SystolicBP",
        "DiastolicBP",
        "BMI",
        "HbA1c",
        "FastingGlucose",
    ]

    # Encode labels: Low -> 0, Mid -> 1, High -> 2
    label_map = {
        "low risk": 0, "low": 0, 0: 0,
        "mid risk": 1, "mid": 1, "medium risk": 1, "medium": 1, 1: 1,
        "high risk": 2, "high": 2, 2: 2,
    }
    df["target"] = df["RiskLevel"].apply(
        lambda x: label_map[str(x).strip().lower()] if isinstance(x, str) else label_map[x]
    )

    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    # Stratified 70:15:15 split (4,272 train / 915 val / 916 test for 6,103 rows)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    # Fit MinMaxScaler strictly on training data to prevent data leakage
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Serialize 8 static features into 8 one-dimensional steps: (N, 8, 1)
    X_train_seq = X_train_scaled.reshape(-1, 8, 1)
    X_val_seq = X_val_scaled.reshape(-1, 8, 1)
    X_test_seq = X_test_scaled.reshape(-1, 8, 1)

    return (X_train_seq, y_train), (X_val_seq, y_val), (X_test_seq, y_test), scaler


# 4. Experimental LSTM Architecture (Section 4.2.4.1 & Section 5.1.4)
class MaternalRiskLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, dropout_rate=0.20, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=8, input_size=1)
        lstm_out, _ = self.lstm(x)
        # Extract the final timestep hidden state: (batch_size, hidden_size=50)
        last_step_out = lstm_out[:, -1, :]
        dropped = self.dropout(last_step_out)
        logits = self.fc(dropped)
        return logits


# 5. Training Loop with Inverse-Frequency Class Weighting & Early Stopping
def train_and_evaluate():
    df = fetch_mendeley_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = prepare_data(df)

    print(f"Dataset Splits -> Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Create PyTorch DataLoaders (Batch size 1,024 as reported in Section 5.1.4)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=1024,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=1024,
        shuffle=False,
    )

    # Inverse-frequency class weights from training partition
    class_counts = np.bincount(y_train)
    inv_freq_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(inv_freq_weights, dtype=torch.float32).to(device)

    model = MaternalRiskLSTM(
        input_size=1, hidden_size=50, dropout_rate=0.20, num_classes=3
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    max_epochs = 180
    patience = 15
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(y_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(y_val)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\nBest Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    model.load_state_dict(best_weights)

    # 6. Held-Out Test Set Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test).to(device))
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=1)

    acc = accuracy_score(y_test, test_preds) * 100
    macro_prec = precision_score(y_test, test_preds, average="macro")
    macro_rec = recall_score(y_test, test_preds, average="macro")
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    macro_auc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")

    # Multiclass Brier Score: mean squared error across class probability vectors
    y_test_onehot = np.eye(3)[y_test]
    brier_score = np.mean(np.sum((test_probs - y_test_onehot) ** 2, axis=1))

    print("\n--- Held-Out Test Set Results ---")
    print(f"Accuracy:        {acc:.1f}%")
    print(f"Macro Precision: {macro_prec:.3f}")
    print(f"Macro Recall:    {macro_rec:.3f}")
    print(f"Macro F1-Score:  {macro_f1:.3f}")
    print(f"Macro ROC-AUC:   {macro_auc:.3f}")
    print(f"Brier Score:     {brier_score:.3f}")

    print("\n--- Per-Class Classification Report ---")
    target_names = ["Low risk", "Mid risk", "High risk"]
    print(classification_report(y_test, test_preds, target_names=target_names, digits=3))

    print("--- Confusion Matrix (Rows: Actual, Cols: Predicted) ---")
    print(confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    train_and_evaluate()
