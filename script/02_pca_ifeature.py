import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split

# ==== 设置 CUDA 设备 ====
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== 加载 ESM 嵌入 ====
embedding_file = '/mnt/data/lzp/OGT/output/protein_embeddings.pt'
X_embeddings = torch.load(embedding_file)

if isinstance(X_embeddings, list):
    X_embeddings = [torch.tensor(embed) for embed in X_embeddings]
    X_esm = torch.cat(X_embeddings, dim=0).numpy()
else:
    X_esm = X_embeddings.numpy()

# ==== 加载并处理理化特征 ====
protein_features_file = '/mnt/data/lzp/OGT/result/ifeature_result/protein_features.csv'
protein_features_df = pd.read_csv(protein_features_file, header=1)  # 从第二行开始读
protein_features = protein_features_df.iloc[:, 1:]  # 删除第一列
protein_features = protein_features.iloc[:X_esm.shape[0], :]  # 对齐行数

# 标准化 + PCA 降维
scaler = StandardScaler()
physico_scaled = scaler.fit_transform(protein_features.values)

pca = PCA(n_components=256)
physico_pca = pca.fit_transform(physico_scaled)

# ==== 特征融合 ====
X_combined = np.concatenate([X_esm, physico_pca], axis=1)
print("X_combined shape:", X_combined.shape)

# ==== 加载标签 ====
labels_file = '/mnt/data/lzp/OGT/output/protein_labels.csv'
y = pd.read_csv(labels_file)['ogt'].values

# ==== 划分数据集 ====
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# ==== 模型定义 ====
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super().__init__()
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
    def __init__(self, patience=15, min_delta=1e-4):
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

# ==== 准备 Tensor ====
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

# ==== Optuna 调参 ====
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

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_losses.append(criterion(model(xb), yb).item())
    return np.mean(val_losses)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("最佳超参数:", best_params)

# ==== 最终训练模型 ====
model = DNNModel(X_train.shape[1], best_params['hidden_dim1'], best_params['hidden_dim2'], best_params['dropout_rate']).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
criterion = nn.MSELoss()
early_stopper = EarlyStopping()

train_losses, val_losses = [], []

for epoch in range(200):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
    early_stopper(val_losses[-1])
    if early_stopper.early_stop:
        print("早停触发，提前终止训练")
        break

# ==== 测试集评估 ====
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).cpu().numpy()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n最终模型 MSE: {mse:.2f}")
print(f"最终模型 R²: {r2:.2f}")

# ==== 保存结果 ====
result_dir = '/mnt/data/lzp/OGT/result/02_ogt_esm_pca_physico'
os.makedirs(result_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))

with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"R²: {r2:.2f}\n")

# ==== 可视化 ====
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.grid()
plt.savefig(os.path.join(result_dir, 'prediction_vs_actual.png'))
plt.show()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()
plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
plt.show()
