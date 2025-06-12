import yfinance as yf
import indicator
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date, period='1d', pre_days=300):

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        extended_start_dt = start_dt - timedelta(days=pre_days)
        extended_start = extended_start_dt.strftime("%Y-%m-%d")

        extended_full_data = yf.download(ticker, start=extended_start, end=end_date, interval=period)
        extended_full_data.columns = extended_full_data.columns.droplevel('Ticker')
        close_prices = extended_full_data.loc[start_date:end_date]['Close'].squeeze()
        extended_prices = extended_full_data.loc[extended_start:end_date]['Close'].squeeze()
        volume = extended_full_data.loc[start_date:end_date]['Volume'].squeeze()
        extended_volume = extended_full_data.loc[extended_start:end_date]['Volume'].squeeze()
        full_data = extended_full_data.loc[start_date:end_date]

        if extended_full_data.empty:
            raise ValueError("未獲取到數據，請檢查股票代碼和日期範圍")

        return full_data, extended_full_data, close_prices, extended_prices, volume, extended_volume

    except Exception as e:
        print(f"數據獲取失敗: {str(e)}")

        return None, None, None, None, None, None

def VIX(start_date, end_date):
    # 以 Yahoo Finance 上的 VIX 代碼建立一個 Ticker 物件
    VIX = yf.download("^VIX", start_date, end_date)
    VIX = VIX['Close']

    return VIX

if __name__ == "__main__":

    ticker = 'QQQ'
    start_date = "2021-1-1"
    end_date = "2025-6-10"

    full_data, extended_full_data, close_prices, extended_prices, volume, extended_volume = get_stock_data(ticker, start_date, end_date)
    rsi = indicator.RSI(close_prices, extended_prices, [7, 14])
    pct_change = indicator.pct_change(close_prices)
    vol_pct_change = indicator.vol_pct_change(volume)
    macd = indicator.MACD(close_prices, extended_prices)
    vix = VIX(start_date,end_date)
    k_line = indicator.K_line(full_data)

    indicator_df = pd.concat([full_data, pct_change, k_line, vol_pct_change, rsi, macd, vix], axis=1)
    indicator_df = indicator_df.round(4)
    indicator_df.to_csv('indicator.csv')
