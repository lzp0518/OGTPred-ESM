import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

# ========== 设置设备 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== 加载数据 ==========
embedding_file = '/mnt/data/lzp/OGT/output/protein_embeddings.pt'
labels_file = '/mnt/data/lzp/OGT/output/protein_labels.csv'

X_embeddings = torch.load(embedding_file)
if isinstance(X_embeddings, list):
    X_embeddings = [torch.tensor(embed) for embed in X_embeddings]
    X = torch.cat(X_embeddings, dim=0).numpy()
else:
    X = X_embeddings.numpy()

y = pd.read_csv(labels_file)['ogt'].values

# ========== 按温度分组 ==========
def classify_ogt(temp):
    if temp < 30:
        return 0
    elif temp <= 60:
        return 1
    else:
        return 2

labels = np.array([classify_ogt(t) for t in y])
X_groups = [X[labels == i] for i in range(3)]
y_groups = [y[labels == i] for i in range(3)]

# ========== 定义模型 ==========
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ========== 模型训练函数 ==========
def train_regression_model(X_data, y_data, root_result_dir, tag):
    result_dir = os.path.join(root_result_dir, tag)
    os.makedirs(result_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    full_train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int(0.8 * len(full_train_data))
    val_size = len(full_train_data) - train_size
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    def objective(trial):
        hidden_dim1 = trial.suggest_int('hidden_dim1', 256, 1024, step=64)
        hidden_dim2 = trial.suggest_int('hidden_dim2', 128, 512, step=32)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        model = DNNModel(X_train.shape[1], hidden_dim1, hidden_dim2, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        for _ in range(30):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        val_losses = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        return np.mean(val_losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    params = study.best_params

    with open(os.path.join(result_dir, 'best_params.txt'), 'w') as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

    model = DNNModel(X_train.shape[1], params['hidden_dim1'], params['hidden_dim2'], params['dropout_rate']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=10)

    train_losses, val_losses = [], []
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"[{tag}] Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
        stopper(val_losses[-1])
        if stopper.early_stop:
            print(f"[{tag}] 早停触发")
            break

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device)).cpu().numpy()
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[{tag}] MSE: {mse:.2f} | R²: {r2:.2f}")

    torch.save(model.state_dict(), os.path.join(result_dir, f'{tag}_model.pth'))
    with open(os.path.join(result_dir, f'{tag}_metrics.txt'), 'w') as f:
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"R²: {r2:.2f}\n")

    plt.figure()
    plt.scatter(y_test, preds, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{tag}: Actual vs Predicted')
    plt.grid()
    plt.savefig(os.path.join(result_dir, f'{tag}_prediction.png'))
    plt.close()

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{tag}: Train vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, f'{tag}_loss_curve.png'))
    plt.close()

    return model

# ========== 执行训练 ==========
result_dir = '/mnt/data/lzp/OGT/result/ogt_regression_by_class'
model_low = train_regression_model(X_groups[0], y_groups[0], result_dir, 'low')
model_mid = train_regression_model(X_groups[1], y_groups[1], result_dir, 'mid')
model_high = train_regression_model(X_groups[2], y_groups[2], result_dir, 'high')
