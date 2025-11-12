import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 设置设备 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==== 加载数据 ====
embedding_file = '/mnt/data/lzp/OGT/output/protein_embeddings.pt'
X_embeddings = torch.load(embedding_file)

if isinstance(X_embeddings, list):
    X_embeddings = [torch.tensor(embed) for embed in X_embeddings]
    X = torch.cat(X_embeddings, dim=0).numpy()
else:
    X = X_embeddings.numpy()

labels_file = '/mnt/data/lzp/OGT/output/protein_labels.csv'
ogt_raw = pd.read_csv(labels_file)['ogt'].values

# ==== 标签分为3类 ====
def classify_ogt(temp):
    if temp < 30:
        return 0  # 低温
    elif temp <= 60:
        return 1  # 中温
    else:
        return 2  # 高温

y = np.array([classify_ogt(t) for t in ogt_raw])

# ==== 划分训练测试 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 定义分类模型 ====
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 3)  # 输出3类
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

# ==== EarlyStopping 类 ====
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

# ==== 转为 Tensor ====
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ==== Dataset 与 Dataloader ====
full_train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# ==== Optuna 调参目标函数 ====
def objective(trial):
    hidden_dim1 = trial.suggest_int('hidden_dim1', 256, 1024, step=64)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 128, 512, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    model = DNNModel(X_train.shape[1], hidden_dim1, hidden_dim2, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(30):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
    return np.mean(val_losses)

# ==== 超参数搜索 ====
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("最佳超参数:", best_params)

# ==== 最终模型训练 + EarlyStopping ====
best_model = DNNModel(
    input_dim=X_train.shape[1],
    hidden_dim1=best_params['hidden_dim1'],
    hidden_dim2=best_params['hidden_dim2'],
    dropout_rate=best_params['dropout_rate']
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
early_stopper = EarlyStopping(patience=15)

train_losses, val_losses = [], []

for epoch in range(200):
    best_model.train()
    epoch_train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    best_model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
    val_losses.append(epoch_val_loss / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    early_stopper(val_losses[-1])
    if early_stopper.early_stop:
        print("早停触发，提前终止训练")
        break

# ==== 测试集预测 ====
best_model.eval()
with torch.no_grad():
    logits = best_model(X_test_tensor.to(device))
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()

# ==== 评估 ====
report = classification_report(y_test, y_pred, target_names=['Low', 'Mid', 'High'])
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\n分类报告:\n", report)
print("混淆矩阵:\n", conf_matrix)
print("准确率: {:.2f}%".format(accuracy * 100))

# ==== 保存结果 ====
result_dir = '/mnt/data/lzp/OGT/result/03_ogt_CLASSIFICATION'
os.makedirs(result_dir, exist_ok=True)

torch.save(best_model.state_dict(), os.path.join(result_dir, 'best_model.pth'))

with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
    f.write(report)
    f.write("\nAccuracy: {:.2f}%\n".format(accuracy * 100))

# ==== 可视化混淆矩阵 ====
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Mid', 'High'],
            yticklabels=['Low', 'Mid', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
plt.show()

# ==== 可视化训练过程 ====
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
