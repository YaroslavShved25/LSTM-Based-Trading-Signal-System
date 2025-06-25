import pandas as pd
from ta.trend import MACD, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands

def engineer_features(df):
    df = df.copy()  # Work on a separate copy to avoid modifying the original reference

    # === 🔁 Basic Technical Features ===

    # Daily return as a percentage change from the previous close
    df['returns'] = df['time_series_close'].pct_change()

    # Short-term rolling volatility: standard deviation of returns (5-day window)
    df['volatility'] = df['returns'].rolling(window=5).std()

    # 5-day moving average of trading volume
    df['volume_ma'] = df['time_series_volume'].rolling(window=5).mean()

    # === 📈 RSI: Relative Strength Index ===

    # Ensure RSI values are numeric and check for overbought condition (RSI > 70)
    df['rsi_rsi'] = pd.to_numeric(df['rsi_rsi'], errors='coerce')
    df['rsi_ob'] = (df['rsi_rsi'] > 70).astype(int)

    # === 📊 MACD: Moving Average Convergence Divergence ===

    macd = MACD(df['time_series_close'])  # Initialize MACD indicator
    df['macd'] = macd.macd().shift(1)         # MACD line (shifted for next-day prediction)
    df['macd_diff'] = macd.macd_diff().shift(1)  # MACD histogram (signal strength)

    # === 📉 ADX: Average Directional Index ===

    adx = ADXIndicator(
        high=df['time_series_high'],
        low=df['time_series_low'],
        close=df['time_series_close'],
        window=10
    )
    df['adx'] = adx.adx().shift(1)            # Trend strength
    df['di_plus'] = adx.adx_pos().shift(1)    # Positive directional index
    df['di_minus'] = adx.adx_neg().shift(1)   # Negative directional index

    # === 📊 Stochastic Oscillator ===

    stoch = StochasticOscillator(
        high=df['time_series_high'],
        low=df['time_series_low'],
        close=df['time_series_close']
    )
    df['stoch_k'] = stoch.stoch().shift(1)            # %K line
    df['stoch_d'] = stoch.stoch_signal().shift(1)     # %D line (signal)

    # === 🎯 Bollinger Bands Position ===

    bb = BollingerBands(df['time_series_close'])
    bb_upper = bb.bollinger_hband().shift(1)
    bb_lower = bb.bollinger_lband().shift(1)

    # Normalized position within Bollinger Bands (0 = lower band, 1 = upper band)
    df['bb_pos'] = ((df['time_series_close'] - bb_lower) / (bb_upper - bb_lower)).clip(0, 1)

    # === 📐 Distance from EMA ===

    # Assumes 'ema_ema' (exponential moving average) is already in the DataFrame
    df['ema_distance'] = (df['time_series_close'] - df['ema_ema']).shift(1)

    # === 🧮 Additional Statistical Features ===

    # Recalculate returns with higher window
    df['returns'] = df['time_series_close'].pct_change()

    # 10-day volatility (standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=10).std()

    # 10-day moving average of volume
    df['volume_ma'] = df['time_series_volume'].rolling(window=10).mean()

    # Price vs. 10-day moving average (price position indicator)
    df['price_vs_ma'] = df['time_series_close'] / df['time_series_close'].rolling(10).mean()

    # Volume spike: current volume vs. average volume
    df['volume_spike'] = df['time_series_volume'] / df['time_series_volume'].rolling(10).mean()

    # Price changes over different time horizons
    df['price_change_5'] = df['time_series_close'].pct_change(5)
    df['price_change_15'] = df['time_series_close'].pct_change(15)

    # Volatility over 5-day window (short-term)
    df['volatility_5'] = df['returns'].rolling(5).std()

    # Drop any rows with NaN values caused by rolling windows or shifting
    return df.dropna()