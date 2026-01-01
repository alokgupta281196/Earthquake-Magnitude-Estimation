# =====================================================
# Comparative Study of ML and DL Models
# Earthquake Magnitude Estimation
# Random Forest vs XGBoost vs GRU | CPU vs GPU
# =====================================================

import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
import joblib
import os

# ==========================================
# 1. LOAD DATA
# ==========================================
CSV_FILE = "final.csv"

df = pd.read_csv(CSV_FILE, low_memory=False)

print("Raw data shape:", df.shape)

# ==========================================
# 2. STANDARDIZE COLUMN NAMES (OPTIONAL BUT GOOD)
# ==========================================
df.columns = df.columns.str.lower().str.strip()

# ==========================================
#  KEEP ONLY IMPORTANT COLUMNS
# ==========================================
required_cols = [
    "time", "latitude", "longitude", "depth", "mag", "place"
]
df = df[required_cols]

# ==========================================
#  CONVERT NUMERIC COLUMNS SAFELY
# ==========================================
num_cols = ["latitude", "longitude", "depth", "mag"]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
#  CLEAN TIME COLUMN
# ==========================================
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# ==========================================
#  REMOVE INVALID / MISSING DATA
# ==========================================
df.dropna(inplace=True)

# ==========================================
#  REMOVE DUPLICATES
# ==========================================
df.drop_duplicates(inplace=True)

# ==========================================
#  REMOVE IMPOSSIBLE VALUES (QUALITY CHECK)
# ==========================================
df = df[
    (df["mag"] >= 0) &
    (df["depth"] >= 0) &
    (df["latitude"].between(-90, 90)) &
    (df["longitude"].between(-180, 180))
]

# ==========================================
#  SORT BY TIME (CRITICAL FOR GRU/LSTM)
# ==========================================
df.sort_values("time", inplace=True)

# ==========================================
# RESET INDEX
# ==========================================
df.reset_index(drop=True, inplace=True)

print("Cleaned data shape:", df.shape)
print(df.head())

# =====================================================
# FEATURE SCALING
# =====================================================
features = ['latitude', 'longitude', 'depth', 'mag']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# =====================================================
# 3. CREATE TIME SERIES SEQUENCES
# =====================================================
def create_sequences(data, seq_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len, :-1])
        y.append(data[i + seq_len, -1])
    return np.array(X), np.array(y)

SEQ_LEN = 5
X, y = create_sequences(scaled, SEQ_LEN)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# =====================================================
# 4. RANDOM FOREST MODEL
# =====================================================
print("\nTraining Random Forest...")
rf_start = time.time()

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_flat, y_train)
rf_preds = rf.predict(X_test_flat)

rf_time = time.time() - rf_start

# Inverse scaling
dummy_rf = np.zeros((len(rf_preds), 4))
dummy_rf[:, -1] = rf_preds
rf_real_preds = scaler.inverse_transform(dummy_rf)[:, -1]

# =====================================================
# 5. XGBOOST MODEL (NEW)
# =====================================================
print("\nTraining XGBoost...")
xgb_start = time.time()

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42
)

xgb.fit(X_train_flat, y_train)
xgb_preds = xgb.predict(X_test_flat)

xgb_time = time.time() - xgb_start

# Inverse scaling
dummy_xgb = np.zeros((len(xgb_preds), 4))
dummy_xgb[:, -1] = xgb_preds
xgb_real_preds = scaler.inverse_transform(dummy_xgb)[:, -1]

# =====================================================
# 6. GRU MODEL
# =====================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

def train_gru(device):
    print(f"Training GRU on {device}")

    model = GRUModel(3, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Convert to tensors (CPU first)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    # Create DataLoader (IMPORTANT)
    batch_size = 1024 if device.type == "cuda" else 2048

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True
    )

    epochs = 60 if device.type == "cuda" else 40
    start = time.time()

    for _ in trange(epochs, desc=f"GRU ({device.type.upper()})"):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(Xte.to(device)).cpu().numpy()

    return preds, train_time


# =====================================================
# 7. GRU CPU & GPU
# =====================================================
gru_cpu_preds, cpu_time = train_gru(torch.device("cpu"))

if torch.cuda.is_available():
    gru_gpu_preds, gpu_time = train_gru(torch.device("cuda"))
    best_gru_preds = gru_gpu_preds
else:
    gpu_time = None
    best_gru_preds = gru_cpu_preds

# Inverse scaling GRU
dummy_gru = np.zeros((len(best_gru_preds), 4))
dummy_gru[:, -1] = best_gru_preds
gru_real_preds = scaler.inverse_transform(dummy_gru)[:, -1]

dummy_actual = np.zeros((len(y_test), 4))
dummy_actual[:, -1] = y_test
real_actual = scaler.inverse_transform(dummy_actual)[:, -1]

# =====================================================
# 8. METRICS (REAL MAGNITUDE)
# =====================================================
rf_rmse = np.sqrt(mean_squared_error(real_actual, rf_real_preds))
rf_mae = mean_absolute_error(real_actual, rf_real_preds)

xgb_rmse = np.sqrt(mean_squared_error(real_actual, xgb_real_preds))
xgb_mae = mean_absolute_error(real_actual, xgb_real_preds)

gru_rmse = np.sqrt(mean_squared_error(real_actual, gru_real_preds))
gru_mae = mean_absolute_error(real_actual, gru_real_preds)

# =====================================================
# 9. RESULTS
# =====================================================
print("\n==================== RESULTS ====================")

print(
    f"Random Forest | "
    f"RMSE: {rf_rmse:.4f} | "
    f"MAE: {rf_mae:.4f} | "
    f"Training Time (CPU): {rf_time:.2f}s"
)

print(
    f"XGBoost       | "
    f"RMSE: {xgb_rmse:.4f} | "
    f"MAE: {xgb_mae:.4f} | "
    f"Training Time (CPU): {xgb_time:.2f}s"
)

print(
    f"GRU (CPU)     | "
    f"RMSE: {gru_rmse:.4f} | "
    f"MAE: {gru_mae:.4f} | "
    f"Training Time: {cpu_time:.2f}s"
)

if gpu_time is not None:
    print(
        f"GRU (GPU)     | "
        f"RMSE: {gru_rmse:.4f} | "
        f"MAE: {gru_mae:.4f} | "
        f"Training Time: {gpu_time:.2f}s"
    )
    print(f"Speedup (CPU → GPU): {cpu_time / gpu_time:.2f}x")
else:
    print("GRU (GPU)     | Not available (CUDA not detected)")


# =====================================================
# 10. SAMPLE OUTPUT
# =====================================================
print("\nSample Prediction:")
print(f"Actual Magnitude   : {real_actual[0]:.2f}")
print(f"RF Prediction      : {rf_real_preds[0]:.2f}")
print(f"XGB Prediction     : {xgb_real_preds[0]:.2f}")
print(f"GRU Prediction     : {gru_real_preds[0]:.2f}")



# =====================================================
# 11. PLOT COMPARISON
# =====================================================
plt.figure(figsize=(12, 5))
plt.plot(real_actual[:300], label="Actual", linewidth=3)
plt.plot(rf_real_preds[:300], label="Random Forest")
plt.plot(xgb_real_preds[:300], label="XGBoost")
plt.plot(gru_real_preds[:300], label="GRU")
plt.title("Actual vs Predicted Earthquake Magnitudes (Time Series)")
plt.xlabel("Samples")
plt.ylabel("Magnitude (Mw)")
plt.legend()
plt.grid(True)
# 12. Actual vs Predicted Scatter (Best Regression Plot)
plt.figure(figsize=(6, 6))
plt.scatter(real_actual, xgb_real_preds, alpha=0.4)
plt.plot(
    [real_actual.min(), real_actual.max()],
    [real_actual.min(), real_actual.max()],
)
plt.xlabel("Actual Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("Actual vs Predicted Magnitude (XGBoost)")
plt.grid(True)
# 13. Earthquake Magnitude Distribution (EDA)
plt.figure(figsize=(8, 5))
plt.hist(df["mag"], bins=50)
plt.xlabel("Magnitude (Mw)")
plt.ylabel("Frequency")
plt.title("Distribution of Earthquake Magnitudes")
plt.grid(True)
# 14. Earthquakes per Year (Trend Analysis)
df["year"] = df["time"].dt.year
yearly_counts = df["year"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.plot(yearly_counts.index, yearly_counts.values)
plt.xlabel("Year")
plt.ylabel("Number of Earthquakes")
plt.title("Yearly Earthquake Frequency")
plt.grid(True)

# ==============================
# Save ML models
# ==============================
print(">>> Saving models now...........")
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf.pkl")
joblib.dump(xgb, "models/xgb.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save GRU model
torch.save(best_gru_preds, "models/gru_preds.npy")
print("Models saved successfully !!!!!!!!!!!")

plt.show(block=True)

