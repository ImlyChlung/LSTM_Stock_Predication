import torch.nn as nn

# 定義 LSTM 模型，輸出層為三分類：
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        # 輸出層為3維對應三分類
        self.fc = nn.Linear(hidden_size, 3)
        # 使用 CrossEntropyLoss 時，不需要額外的激活函數（比如 softmax ）

    def forward(self, x):
        out, _ = self.lstm(x)
        # 取序列最後一個時間步的輸出
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)  # logits 輸出
        return out