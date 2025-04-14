# Stock Price Prediction with TensorFlow & Alpha Vantage

This project uses a trained LSTM deep learning model to predict stock prices based on historical data fetched via the Alpha Vantage API. It includes real-time prediction, evaluation metrics, and 3-day + 1-hour forecasts.

## 🚀 Features
- Predicts stock prices using a pre-trained LSTM model.
- Fetches real-time stock data using the Alpha Vantage API.
- Plots Actual vs Predicted prices.
- Predicts the next 3 days and next 1 hour (if minute data available).
- Calculates MAE, MSE, RMSE, and R² Score for performance analysis.

## 📊 Model Evaluation Example
```
📏 MAE: 21.1776
📏 MSE: 2411.1495
📏 RMSE: 49.1035
📏 R² Score: 0.9698
```

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- Alpha Vantage API
- Matplotlib & Seaborn
- Pandas / NumPy
- Scikit-learn

## 🧠 Model Info
- Type: LSTM (Long Short-Term Memory)
- Trained on: 60-day time windows of historical daily stock prices
- Input: Stock Ticker
- Output: Forecasts and performance metrics

## 📦 Requirements
- Python 3.7+
- TensorFlow
- Alpha Vantage (`pip install alpha_vantage`)
- matplotlib, seaborn, pandas, sklearn, etc.

## 🧪 How to Use
1. Clone the repo and install requirements.
2. Replace `ALPHA_VANTAGE_API_KEY` with your key.
3. Place the trained `.h5` model file at the given path.
4. Run `TESTING.py` and enter a stock ticker when prompted.

## 📝 License
MIT License

## 💡 Credits
Created by Deeraj (2025). Built for experimentation and showcasing ML in stock forecasting.
