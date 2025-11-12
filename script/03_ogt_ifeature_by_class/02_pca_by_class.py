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

# ==== 加载数据 ====
embedding_file = '/mnt/data/lzp/OGT/output/protein_embeddings.pt'
X_embeddings = torch.load(embedding_file)
if isinstance(X_embeddings, list):
    X_embeddings = [torch.tensor(embed) for embed in X_embeddings]
    X_esm = torch.cat(X_embeddings, dim=0).numpy()
else:
    X_esm = X_embeddings.numpy()

protein_features_file = '/mnt/data/lzp/OGT/result/ifeature_result/protein_features.csv'
protein_features_df = pd.read_csv(protein_features_file, header=1)
protein_features = protein_features_df.iloc[:, 1:]
protein_features = protein_features.iloc[:X_esm.shape[0], :]

# 标准化 + PCA
scaler = StandardScaler()
physico_scaled = scaler.fit_transform(protein_features.values)
pca = PCA(n_components=256)
physico_pca = pca.fit_transform(physico_scaled)

X_combined = np.concatenate([X_esm, physico_pca], axis=1)
print("X_combined shape:", X_combined.shape)

labels_file = '/mnt/data/lzp/OGT/output/protein_labels.csv'
y = pd.read_csv(labels_file)['ogt'].values

# ==== 按照OGT分组 ====
bins = [-np.inf, 30, 60, np.inf]
group_names = ['low', 'medium', 'high']
group_labels = pd.cut(y, bins=bins, labels=group_names)
group_labels = np.array(group_labels)

print("每组样本数量：")
print(pd.Series(group_labels).value_counts())

# ==== 定义模型 ====
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

# ==== 训练和评估函数 ====
def train_and_evaluate(X, y, group_name):
    print(f"\n===== 开始训练组：{group_name}，样本数：{len(y)} =====")

    # 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # TensorDataset
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

    # Optuna超参数调优
    def objective(trial):
        hidden_dim1 = trial.suggest_int('hidden_dim1', 256, 1024, step=64)
        hidden_dim2 = trial.suggest_int('hidden_dim2', 128, 512, step=32)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        model = DNNModel(X.shape[1], hidden_dim1, hidden_dim2, dropout_rate).to(device)
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
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"{group_name}组最优超参数:", best_params)

    # 使用最优超参数训练最终模型
    model = DNNModel(X.shape[1], best_params['hidden_dim1'], best_params['hidden_dim2'], best_params['dropout_rate']).to(device)
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

    # 测试集评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.to(device)).cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{group_name}组测试集 MSE: {mse:.2f}, R²: {r2:.2f}")

    # 保存模型和结果
    result_dir = f'/mnt/data/lzp/OGT/result/02_ogt_esm_pca_physico_by_class/{group_name}'
    os.makedirs(result_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pth'))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"R²: {r2:.2f}\n")

    # 画图保存
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted ({group_name})')
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'prediction_vs_actual.png'))
    plt.close()

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss ({group_name})')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    plt.close()

# ==== 主流程，循环三组分别训练 ====
for g in group_names:
    idx = np.where(group_labels == g)[0]
    train_and_evaluate(X_combined[idx], y[idx], g)
