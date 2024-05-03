import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ccxt

# Fetch historical cryptocurrency data
def fetch_data(symbol, timeframe, limit):
    url = f"https://api.binance.com/api/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    queryes = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Technical analysis indicators
def add_indicators(df):
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['close'])
    return df

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Generate trading signals
def generate_signals(df):
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['close']
    signals['signal'] = 0
    signals['signal'][50:] = np.where(df['SMA_50'][50:] > df['SMA_200'][50:], 1, 0)
    signals['signal'][(df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)] = 1
    signals['signal'][(df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)] = 0
    return signals

# Backtest trading strategy
def backtest(signals):
    signals['positions'] = signals['signal'].diff()
    initial_capital = 10000
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['crypto'] = 100 / signals['price']
    portfolio = positions.multiply(signals['price'], axis=0)
    pos_diff = signals['signal'].diff()
    portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    return portfolio

# Main function
def main():
    symbol = 'BTC/USDT'
    timeframe = '1d'
    limit = 1000
    data = fetch_data(symbol, timeframe, limit)
    data = add_indicators(data)
    signals = generate_signals(data)
    portfolio = backtest(signals)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax[0].plot(data['close'], label='Close Price')
    ax[0].plot(data['SMA_50'], label='SMA 50')
    ax[0].plot(data['SMA_200'], label='SMA 200')
    ax[0].legend()

    ax[1].plot(signals.loc[signals['signal'] == 1].index, signals['price'][signals['signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    ax[1].plot(signals.loc[signals['signal'] == 0].index, signals['price'][signals['signal'] == 0], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    ax[1].plot(portfolio['total'], label='Portfolio Value', color='b')
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    main()
