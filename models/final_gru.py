import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import warnings
import gc

warnings.filterwarnings('ignore')

print(" Starting the 'Final Boss' GRU Pipeline...")


# 1. SETUP & HYPERPARAMETERS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Hyperparameters ---
WINDOW_SIZE = 30         # Use 30 minutes of history
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
EPOCHS = 25              # NNs learn fast, we don't need too many epochs
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4            # Heavy regularization to fight noise

# 2. FEATURE ENGINEERING (Z-Score + Context)

print("⚙️ Engineering Context-Aware Features...")
train_df = pd.read_csv('dataset/train_v2.csv').sort_values(['date_id', 'minute_id'])
test_df = pd.read_csv('dataset/test_v2.csv').sort_values(['date_id', 'minute_id'])

# Combine for efficient feature creation
len_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)
combined.replace([np.inf, -np.inf], np.nan, inplace=True)

feature_cols = []
base_feats = [f'feature_{i}' for i in range(1, 31)]

for col in base_feats:
    # Daily Mean and Std for Z-Score and Context
    daily_mean = combined.groupby('date_id')[col].transform('mean')
    daily_std = combined.groupby('date_id')[col].transform('std')
    
    # Z-Score (The Signal)
    combined[f'{col}_z'] = (combined[col] - daily_mean) / (daily_std + 1e-5)
    feature_cols.append(f'{col}_z')
    
    # Daily Volatility (The Context)
    combined[f'{col}_std'] = daily_std
    feature_cols.append(f'{col}_std')

# Time Features
combined['minute_sin'] = np.sin(2 * np.pi * combined['minute_id'] / 240.0)
combined['minute_cos'] = np.cos(2 * np.pi * combined['minute_id'] / 240.0)
feature_cols.extend(['minute_sin', 'minute_cos'])

# Fill any remaining NaNs (e.g., from std of a single point)
combined[feature_cols] = combined[feature_cols].fillna(0)

# Split back
train_df = combined.iloc[:len_train]
test_df = combined.iloc[len_train:]
del combined; gc.collect()

print(f" Training on {len(feature_cols)} features.")

# 3. CREATE SLIDING WINDOWS

def create_windows(features, targets, date_ids, window_size):
    X, y = [], []
    for i in range(window_size, len(features)):
        if date_ids[i] == date_ids[i - window_size]:
            X.append(features[i - window_size : i])
            y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

dates = train_df['date_id'].unique()
split_point = int(len(dates) * 0.90) # Train on 90%, Val on 10%
train_date_limit = dates[split_point]

train_set = train_df[train_df['date_id'] <= train_date_limit]
val_set = train_df[train_df['date_id'] > train_date_limit]

print("Creating sliding windows...")
X_train, y_train = create_windows(train_set[feature_cols].values, train_set[['target_short', 'target_medium', 'target_long']].values, train_set['date_id'].values, WINDOW_SIZE)
X_val, y_val = create_windows(val_set[feature_cols].values, val_set[['target_short', 'target_medium', 'target_long']].values, val_set['date_id'].values, WINDOW_SIZE)

# ==========================================
# 4. PYTORCH MODEL & TRAINING
# ==========================================
class ContextGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=3, dropout=0.3):
        super(ContextGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        # We only want the prediction from the very last time step
        last_time_step = gru_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

class WeightedMAELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMAELoss, self).__init__()
        self.register_buffer('weights', torch.tensor(weights, device=DEVICE))
    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        weighted_error = torch.sum(abs_error * self.weights, dim=1)
        return torch.mean(weighted_error)

# DataLoaders
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE)

model = ContextGRU(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=3, dropout=DROPOUT).to(DEVICE)
criterion = WeightedMAELoss(weights=[0.1, 0.3, 0.6])
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

print("\n Training the GRU...")
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            val_loss += criterion(output, target).item()
    
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_gru_model.pth')
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.6f} ⭐ New Best!")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.6f}")

# ==========================================
# 5. INFERENCE & SUBMISSION
# ==========================================
print("\n Predicting on Test Set with GRU...")
model.load_state_dict(torch.load('best_gru_model.pth'))
model.eval()

# Create test windows with padding
X_test_raw = test_df[feature_cols].values
padding = np.tile(X_test_raw[0], (WINDOW_SIZE - 1, 1))
padded_test = np.vstack([padding, X_test_raw])

X_test_windows = []
for i in range(len(test_df)):
    X_test_windows.append(padded_test[i : i + WINDOW_SIZE])
X_test_tensor = torch.tensor(np.array(X_test_windows, dtype=np.float32)).to(DEVICE)

test_loader = DataLoader(X_test_tensor, batch_size=BATCH_SIZE)
final_preds = []
with torch.no_grad():
    for data in test_loader:
        output = model(data)
        final_preds.append(output.cpu().numpy())

final_preds = np.vstack(final_preds)

# Apply post-processing
for i in range(3):
    final_preds[:, i] -= final_preds[:, i].mean()
    final_preds[:, i] *= 0.85 # Gentle shrinkage

submission = pd.DataFrame({
    'id': test_df['id'],
    'target_short': final_preds[:, 0],
    'target_medium': final_preds[:, 1],
    'target_long': final_preds[:, 2]
})

submission.to_csv('submission_jane_street_gru.csv', index=False)
print("\n Final GRU Submission saved as: submission_jane_street_gru.csv")