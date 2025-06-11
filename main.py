import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import LSTM
from data_preprocess import FinanceDataset

def train_and_evaluate(model, dataset, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # 劃分訓練集、驗證集和測試集 (70%訓練, 15%驗證, 15%測試)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    indices = np.arange(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 建立資料載入器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 記錄訓練過程
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    best_model_state = None
    early_stop_patience = 50
    patience_counter = 0

    for epoch in range(epochs):
        # 訓練階段
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # 輸出 shape: (batch, 3)

            # 使用 CrossEntropyLoss 要求 targets shape 為 (batch,)，loss 自動運算 softmax 和 log-softmax
            loss = criterion(outputs, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 基於 logits 用 argmax 求預測的類別
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == targets).sum().item()
            batch_acc = correct / targets.size(0)

            epoch_train_loss += loss.item() * inputs.size(0)
            epoch_train_acc += batch_acc * inputs.size(0)

        # 驗證階段
        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 計算準確率
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == targets).sum().item()
                batch_acc = correct / targets.size(0)

                epoch_val_loss += loss.item() * inputs.size(0)
                epoch_val_acc += batch_acc * inputs.size(0)

        # 計算平均指標
        epoch_train_loss /= len(train_dataset)
        epoch_train_acc /= len(train_dataset)
        epoch_val_loss /= len(val_dataset)
        epoch_val_acc /= len(val_dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        # 學習率調整
        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 載入最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation accuracy")

    # 在測試集上評估
    model.eval()
    all_preds, all_targets = [], []
    test_loss, test_acc = 0.0, 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == targets).sum().item()
            batch_acc = correct / targets.size(0)

            test_loss += loss.item() * inputs.size(0)
            test_acc += batch_acc * inputs.size(0)

            # 將預測和目標轉換成 numpy 整型數據
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_dataset)
    test_acc /= len(test_dataset)

    print(f"\nTest Results: Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    # 繪製訓練過程
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 輸出詳細評估報告
    pred_labels = np.array(all_preds)
    true_labels = np.array(all_targets)

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    # 繪製預測分佈
    plt.figure(figsize=(10, 6))
    plt.hist(all_preds, bins=50, alpha=0.7, color='blue')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2)
    plt.title("Prediction Distribution on Test Set")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 返回訓練好的模型和測試結果
    return model, {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': all_preds,
        'targets': all_targets
    }


def predict_next_day(model, dataset, df, seq_length=20):
    """
     使用訓練好的模型預測下一日的Pct_Change

     參數:
     model: 訓練好的LSTM模型
     dataset: FinanceDataset實例(用於存取scaler)
     df: 完整的原始DataFrame
     seq_length: 序列長度(與訓練時相同)

     返回:
     next_day_pred: 下一天的預測百分比變化
    """
    # 1. 準備最新的序列數據
    # 取得最後seq_length天的數據
    features = ['Pct_Change','Vol_Pct_Change', 'RSI_7', 'RSI_14', 'DIF', 'DEA', 'MACD','upper_shadow_ratio', 'lower_shadow_ratio', 'price_diff_ratio','^VIX']
    latest_data = df[features].tail(seq_length).values

    # 2. 應用相同的歸一化處理
    scaled_data = dataset.scaler.transform(latest_data)

    # 3. 轉換為模型輸入格式
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # (1, seq_length, input_size)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor.to(device))  # logits shape (1,3)
        pred_class = torch.argmax(outputs, dim=1).item()


    return pred_class



# 4. 主程序
if __name__ == "__main__":
    # 参数设置
    SEQ_LENGTH = 30  # 使用30天歷史數據
    INPUT_SIZE = 11  # 特徵數: Pct_Change, upper_shadow_ratio, lower_shadow_ratio, price_diff_ratio, Vol_Pct_Change, RSI_7, RSI_14, DIF, DEA, MACD
    HIDDEN_SIZE = 64  # 隱藏單元大小
    NUM_LAYERS = 2
    EPOCHS = 300
    LEARNING_RATE = 0.001

    # 載入資料集
    print("載入資料集...")
    dataset = FinanceDataset(SEQ_LENGTH)
    print(f"數據集大小: {len(dataset)} 個樣本")

    # 檢查標籤分佈 (統計 3 個類別的頻率)
    labels = dataset.y
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("標籤分佈統計：")
    for label in sorted(label_counts.keys()):
        print(f"  標籤 {label}: {label_counts[label]} 個，佔比 {label_counts[label] / len(labels):.2%}")

    # 初始化模型
    print("初始化模型...")
    model = LSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.3
    )
    print(model)

    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可訓練參數數量: {total_params}")

    # 訓練並評估
    print("開始訓練...")
    trained_model, test_results = train_and_evaluate(
        model, dataset, epochs=EPOCHS, lr=LEARNING_RATE
    )
    print("訓練完成!")

    # 輸出測試結果
    print(f"\n測試集準確率: {test_results['test_acc']:.4f}")
    print(f"測試集損失: {test_results['test_loss']:.4f}")

    # 预测方向：預測函數將傳回 0, 1 或 2
    df = pd.read_csv('indicator.csv', parse_dates=['Date'])

    next_direction_pred = predict_next_day(trained_model, dataset, df, SEQ_LENGTH)

    # 取得最後日期，並推算下一交易日日期（假設交易日連續）
    last_date = df['Date'].iloc[-1]
    next_date = last_date + pd.DateOffset(days=1)

    # 定義分類標籤的映射
    prediction_mapping = {0: "下跌", 1: "平盤/不明確", 2: "上漲"}

    print(f"\n預測結果: {next_date.strftime('%Y-%m-%d')}")
    print(f"預測趨勢為: {prediction_mapping[next_direction_pred]} (標籤: {next_direction_pred})")