

# Stock Price Prediction using LSTM

This repository contains a stock price prediction model built using LSTM (Long Short-Term Memory) and the Alpha Vantage API for fetching historical stock data. The model predicts the future stock prices based on the past data.

##MSFT Achievement
Our cutting-edge LSTM model has achieved outstanding results in forecasting Microsoft (MSFT) stock prices, setting a new standard for accuracy in financial prediction. When we tested the model on Microsoft’s historical data, the performance was nothing short of remarkable:

R² Score: 0.9984 – This near-perfect score reflects an exceptional fit between the predicted and actual stock prices, showing that our model can predict Microsoft’s price movements with unprecedented precision.

MAE: 3.5777, MSE: 18.7211, and RMSE: 4.3268 – These metrics showcase the model’s exceptionally high accuracy, with a negligible margin of error when forecasting stock prices. The model is capable of predicting price movements with impressive reliability, providing valuable insights for both traders and investors.

These results demonstrate the power and accuracy of our LSTM-based approach in predicting the price dynamics of one of the world’s most liquid and volatile stocks—Microsoft. With this model, you can confidently rely on cutting-edge technology to forecast market movements and make data-driven investment decisions with a high degree of certainty. This achievement marks a major milestone in stock price prediction and solidifies our model as a go-to tool for anyone in the financial world.


## Features

- Predicts the stock prices for the next hour based on historical data.
- Evaluates the model's performance using common regression metrics: MAE, MSE, RMSE, and R² Score.
- Visualizes actual vs predicted stock prices in a plot.
- Fetches stock data using the Alpha Vantage API.

## Requirements

- Python 3.6+
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- alpha_vantage
- time

## Installation

1. **Clone the Repository**

   Clone this repository to your local machine using git:

   ```bash
   git clone https://github.com/dhe1raj/TradePredictionModel.git
   ```

2. **Install Dependencies**

   After cloning the repository, navigate to the project directory and install the required libraries by running:

   ```bash
   cd your-repository-name
   pip install -r requirements.txt
   ```

   This will install all the necessary dependencies to run the script.

3. **Set Up Alpha Vantage API Key**

   You will need an **Alpha Vantage API Key** to fetch stock data. Sign up for a free API key [here](https://www.alphavantage.co/support/#api-key).

   After getting your key, open the `TESTING.py` file and replace the placeholder with your API key:

   ```python
   ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'
   ```

4. **Download the Pretrained Model**

   If you don't have a trained model file, you can train it on your own, or use the provided pretrained model `stock_price_model.h5`. Make sure it's placed in the project directory where `TESTING.py` is located.

## Usage

1. **Run the Stock Prediction Script**

   In the terminal, navigate to the project directory and run the `TESTING.py` script:

   ```bash
   python TESTING.py
   ```

2. **Enter Stock Ticker**

   When prompted, enter the stock ticker symbol you wish to predict (e.g., `MSFT` for Microsoft or `TATAMOTORS.NS` for Tata Motors).

   ```
   Enter the stock ticker symbol (e.g., TATAMOTORS.NS or TSLA): MSFT
   ```

3. **View Results**

   The model will:
   - Output evaluation metrics such as MAE, MSE, RMSE, and R² Score.
   - Plot a graph comparing actual vs predicted prices.
   - Predict the next hour's stock price based on the most recent data.

   Example output:
   ```
   📏 MAE: 3.5777, MSE: 18.7211, RMSE: 4.3268, R² Score: 0.9984
   🕒 Short-term (next hour) prediction:
   📈 Predicted next hour price: ₹425.67
   ```

   The plot will be saved as `prediction_plot.pdf`.

## File Structure

- `TESTING.py`: Main script that runs the stock prediction.
- `stock_price_model.h5`: Pretrained model used for prediction.
- `README.md`: Project overview and setup guide.
- `requirements.txt`: List of dependencies for the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Feel free to fork the repository, contribute to the project, and submit pull requests. Please make sure to follow best practices and document your code.
```

### Instructions for Updating:
1. Replace the **repository URL** with your own GitHub repository URL in the first usage section.
2. Replace `'YOUR_API_KEY'` with your actual Alpha Vantage API Key in the script.
3. Ensure that the **model file** (`stock_price_model.h5`) is available in your project directory.
   
This file provides a complete and clear README for your project! Let me know if you need any other edits or changes!
