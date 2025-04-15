import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from datetime import timedelta
import time
from alpha_vantage.timeseries import TimeSeries
import os

# -------- CONFIGURATION --------
MODEL_PATH = r"C:\Users\pc\Desktop\aiprojects\Stock_updated\stock_price_model.h5"
ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'
WINDOW_SIZE = 60

# -------- FETCH DATA --------
def fetch_stock_data(ticker, retries=5):
    attempt = 0
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    while attempt < retries:
        try:
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')  # ✅ Full historical data
            data.dropna(inplace=True)
            data.sort_index(inplace=True)  # ✅ Ensure chronological order (old → new)
            if data.empty:
                raise ValueError("Fetched data is empty. Please try a different time window or ticker.")
            return data
        except (ValueError, Exception) as e:
            print(f"Error fetching data for {ticker}: {str(e)}. Retrying...")
            attempt += 1
            time.sleep(60)
    raise Exception(f"Failed to fetch data for {ticker} after {retries} attempts.")

# -------- PREPROCESSING --------
def preprocess_data(df):
    df = df[['4. close']]
    data = df.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X.append(data_scaled[i - WINDOW_SIZE:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler, df

# -------- PREDICTION UTILS --------
def predict_future(model, last_window, days, scaler):
    predictions = []
    current_input = last_window.copy()
    for _ in range(days):
        pred = model.predict(current_input.reshape(1, WINDOW_SIZE, 1), verbose=0)
        predictions.append(pred[0])
        current_input = np.append(current_input[1:], pred, axis=0)
    return scaler.inverse_transform(predictions)

# -------- EVALUATION --------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📏 MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
    return mae, mse, rmse, r2

# -------- PLOTTING --------
def plot_predictions(dates, actual, predicted, title, export_path="plot.pdf"):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates, y=actual.flatten(), label="Actual", color="blue")
    sns.lineplot(x=dates, y=predicted.flatten(), label="Predicted", color="orange")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(export_path)
    plt.show()

# -------- MAIN ENGINE --------
def run_stock_prediction(ticker):
    print(f"\n📡 Running prediction for: {ticker}")
    model = load_model(MODEL_PATH)

    df = fetch_stock_data(ticker)
    X, y, scaler, df_raw = preprocess_data(df)

    # Predict and evaluate on historical data
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y)
    evaluate_model(y_true, y_pred)

    # Plot actual vs predicted
    plot_predictions(
        df_raw.index[-len(y_true):],
        y_true,
        y_pred,
        title=f"{ticker} - Actual vs Predicted Prices",
        export_path="prediction_plot.pdf"
    )

    # Predict next hour
    print("\n🕒 Short-term (next hour) prediction:")
    last_window = X[-1]
    next_hour = predict_future(model, last_window, days=1, scaler=scaler)
    print(f"📈 Predicted next hour price: ₹{next_hour[-1][0]:.2f}")

    # Predict next 3 days
    print("\n📅 Forecast for Next 3 Days:")
    next_3_days = predict_future(model, last_window, days=3, scaler=scaler)
    future_dates = [df_raw.index[-1] + timedelta(days=i+1) for i in range(3)]
    for i in range(3):
        print(f"📈 Day {i+1} ({future_dates[i].strftime('%Y-%m-%d')}): ₹{next_3_days[i][0]:.2f}")

    plot_predictions(
        future_dates,
        next_3_days,
        next_3_days,
        title=f"{ticker} - 3-Day Forecast",
        export_path="forecast_3_days.pdf"
    )

# -------- CLI RUN --------
if __name__ == "__main__":
    ticker_input = input("Enter the stock ticker symbol (e.g., TATAMOTORS.NS or TSLA): ").strip().upper()
    run_stock_prediction(ticker_input)


