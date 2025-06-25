import os
import joblib
import numpy as np
from load_and_preprocess import load_data
from feature_engineering import engineer_features
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
import matplotlib.pyplot as plt       # For basic plotting and data visualization

SEQ_LEN = 20

# Reshape 2D features into 3D shape for LSTM: [samples, timesteps, features]
def reshape_for_lstm(X, seq_len):
    return np.array([X[i - seq_len:i] for i in range(seq_len, len(X))])

# Build a simple LSTM model for binary classification
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape = input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Output probability
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return model

# Evaluate strategy vs. buy & hold using predicted signals
def backtest_strategy(df, price_col='time_series_close'):
    df = df.copy()
    df = df[df['prediction'].notna()]

    # Calculate price returns
    df['returns'] = df[price_col].pct_change()

    # Strategy returns only when signal == 1
    df['strategy_returns'] = df['signal'].shift(1, fill_value=0) * df['returns']

    # Track portfolio value
    df['strategy_value'] = (1 + df['strategy_returns']).cumprod() * INITIAL_CAPITAL
    df['buy_hold_value'] = (1 + df['returns']).cumprod() * INITIAL_CAPITAL

    # Return dataframe, strategy profit, buy & hold profit, number of trades
    return df, df['strategy_value'].iloc[-1] - INITIAL_CAPITAL, df['buy_hold_value'].iloc[-1] - INITIAL_CAPITAL, df['signal'].diff().abs().sum()

# Plot model predictions vs. actuals
def plot_predictions(index, actual, predicted, ticker):
    plt.figure(figsize=(15, 5))
    plt.plot(index, actual, label='Actual', alpha=0.6)
    plt.plot(index, predicted, label='Predicted Probability', alpha=0.8)
    plt.title(f"{ticker} - LSTM: Actual vs Predicted Probabilities")
    plt.xlabel("Index")
    plt.ylabel("Target / Prob")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot strategy performance vs. buy & hold
def plot_backtest(df, ticker):
    plt.figure(figsize=(15, 4))
    plt.plot(df.index, df['strategy_value'], label='Strategy')
    plt.plot(df.index, df['buy_hold_value'], label='Buy & Hold')
    plt.title(f"{ticker} - Backtest Portfolio Value")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load and train
ticker_dfs = load_data()
os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

for ticker, df in ticker_dfs.items():
    df = df.copy()

    # Define input features (excluding target)
    feature_cols = [col for col in df.columns if col != 'target']

    # Ensure all feature columns are numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=feature_cols + ['target'])

    # Skip tickers with insufficient data
    if len(df) < SEQ_LEN + 50:
        print(f"{ticker}: Skipping due to insufficient data.")
        continue

    # Prepare X and y for LSTM
    X = df[feature_cols].values
    y = df['target'].astype(int).values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM input
    X_lstm = reshape_for_lstm(X_scaled, SEQ_LEN)
    y_lstm = y[SEQ_LEN:]

    # Train/Test split (80/20)
    split = int(len(X_lstm) * 0.8)
    X_train, X_test = X_lstm[:split], X_lstm[split:]
    y_train, y_test = y_lstm[:split], y_lstm[split:]

    # Build and train LSTM model
    model = create_lstm_model((SEQ_LEN, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    start_time = time.time()
    # Predict probabilities and apply threshold
    y_probs = model.predict(X_test).flatten()
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    print(f"⚡ Avg Inference Time per sample: {inference_time:.2f} ms")
    y_pred = (y_probs > THRESHOLD).astype(int)

    # Evaluate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    # Calculate win rate
    wins = ((y_pred == 1) & (y_test == 1)).sum()
    total_buys = (y_pred == 1).sum()
    win_rate = wins / total_buys if total_buys > 0 else 0

    # Output evaluation results
    print(f"\n📊 {ticker} - LSTM Results")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"Win Rate : {win_rate:.2%} ({wins}/{total_buys})")
    print(confusion_matrix(y_test, y_pred))

    # Store predictions and signals in DataFrame
    df['prediction'] = np.nan
    df.iloc[-len(y_test):, df.columns.get_loc('prediction')] = y_probs
    df['signal'] = 0
    df.loc[df['prediction'] > THRESHOLD, 'signal'] = 1

    # Plot predictions
    test_idx = df.iloc[-len(y_test):].index
    plot_predictions(test_idx, y_test, y_probs, ticker)

    # Run and visualize backtest
    test_df, strat_ret, hold_ret, trades = backtest_strategy(df)
    print(f"💰 Backtest for {ticker}")
    print(f"- Strategy Return: ${strat_ret:.2f}")
    print(f"- Buy & Hold     : ${hold_ret:.2f}")
    print(f"- Trades Executed: {int(trades)}")
    plot_backtest(test_df, ticker)
    # Save model and scaler
    model.save(f"models/{ticker.lower()}_lstm.h5")
    joblib.dump(scaler, f"scalers/{ticker.lower()}_scaler.pkl")