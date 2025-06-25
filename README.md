

# 📈 LSTM-Based Trading Signal System

This project uses an **LSTM deep learning model** to generate **buy signals** for high-volatility equities like **$MSTR**, based on historical price and volume data.  
It supports **real-time signal prediction**, **backtesting**, and is **ready for AWS deployment and brokerage automation** via Alpaca.

---

## 🚀 Features
- 🔮 Predicts binary trading signals using an LSTM model
- 💾 Pretrained model and scaler saved for deployment
- 📊 Backtesting engine with strategy vs. buy & hold comparisons
- 📬 Email alerts for manual trade execution (optional)
- 🐳 Containerized with Docker for cloud deployment
- 💹 Brokerage-ready for automated trading (Alpaca supported)

---

## 🧱 Project Structure
```text
lstm-trading-system/
├── data/                # Sample input data (JSON format)
├── models/              # Trained LSTM model and scaler
├── notebooks/           # Jupyter notebooks for research/training
├── src/                 # Source code
│   ├── train_model.py        # Train and save LSTM model
│   ├── predict.py            # Signal prediction & Alpaca integration
│   ├── notify.py             # Email notifications
│   ├── brokerage.py          # Alpaca trade execution (optional)
│   ├── config.py             # Configs and API keys
│   ├── feature_engineering.py# Feature creation functions
│   └── utils.py              # Helper utilities
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
````

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YaroslavShved25/LSTM-Based-Trading-Signal-System.git
cd lstm-trading-system
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file for sensitive keys (Alpaca API, email credentials, etc.)

---

## 🧠 Train the Model

Retrain the model with:

```bash
python src/train_model.py
```

This uses `data/sample_MSTR.json` and saves:

* Model → `models/MSTR_lstm.h5`
* Scaler → `models/MSTR_scaler.pkl`

---

## 🔮 Predict Signals

```python
from src.predict import predict_signals
import pandas as pd

df = pd.read_json("data/sample_MSTR.json")
df_predicted = predict_signals(df, feature_cols=["open", "high", "low", "close", "volume"])
```

---

## 📊 Backtest Performance

```python
from src.backtest import backtest_strategy

df_bt, strat_value, hold_value, trades = backtest_strategy(df_predicted)
```

---

## 📬 Send Alerts (Optional)

```python
from src.notify import send_signal_email

send_signal_email("MSTR", "BUY", 1265.44)
```

---

## ☁️ Docker Deployment

```bash
docker build -t lstm-trader .
docker run --rm --env-file .env lstm-trader
```

---

## 💹 Future: Automated Trading

Once signal accuracy exceeds 80% in backtests, use Alpaca to automate trades:

```python
from src.brokerage import execute_trade

execute_trade("MSTR", "buy", qty=1)
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Questions?

Feel free to reach out or open an issue.
Happy trading! 🚀📈

```

---

