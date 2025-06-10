import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class FinanceDataset(Dataset):
    def __init__(self, seq_length, file_path='indicator.csv'):
        # 讀取數據，並解析日期
        df = pd.read_csv(file_path, parse_dates=['Date'])
        N = 5  # 用作偏移，意即 future_close = closes[i + seq_length + N]


        # 保存日期，用于後續可視化 (注意日期的索引根據序列長度調整)
        self.dates = df['Date'].iloc[seq_length - 1: len(df) - 20].values

        # 提取特徵數據：原有特徵加上新增的 K 線特徵
        features = df[['Pct_Change', 'upper_shadow_ratio', 'lower_shadow_ratio', 'price_diff_ratio', 'Vol_Pct_Change',
                       'RSI_7', 'RSI_14', 'DIF', 'DEA', 'MACD','^VIX']].values

        # 提取收盤價
        closes = df['Close'].values

        self.X, self.y = [], []
        # 調整循環範圍，確保能夠取到未來第N日數據
        for i in range(len(df) - seq_length - N):
            # 輸入序列：取 seq_length 天的特徵
            self.X.append(features[i:i + seq_length])

            # 當前日（序列最後一天）的收盤價
            today_close = closes[i + seq_length - 1]
            # 獲取未來第 (N) 天後的收盤價
            future_close = closes[i + seq_length + N]

            # 計算持有期收益率 (HPR)
            hpr = (future_close - today_close) / today_close

            # 根據收益率劃分三類：
            # HPR > 5% -> 2, HPR < -5% -> 0, 其他 -> 1
            if hpr > 0.03:
                label = 2
            elif hpr < -0.03:
                label = 0
            else:
                label = 1
            self.y.append(label)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # 特徵歸一化：注意需要將三維數據 reshape 為二維，再 reshape 回去
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(
            self.X.reshape(-1, features.shape[1])
        ).reshape(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 使用 long 類型存儲標籤以滿足 CrossEntropyLoss 要求
        return torch.tensor(self.X[idx], dtype=torch.float32), \
            torch.tensor(self.y[idx], dtype=torch.long)
