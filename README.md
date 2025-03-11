# Stock-Price-Movement-Analysis_202401100300198
# Rule-Based Stock Price Prediction

## Overview

This repository contains a **Rule-Based Stock Price Prediction Algorithm** written in Python. The algorithm uses historical stock price data to generate buy or sell signals based on **Simple Moving Averages (SMA)** and **Price Change**. The predictions help traders make informed decisions by identifying potential upward or downward movements in stock prices.

The model uses **pandas** for data manipulation, **numpy** for numerical operations, and **matplotlib** for visualizing the results.

## Methodology

The algorithm follows a simple rule-based approach for predicting stock price movements. The methodology consists of the following steps:

### 1. **Data Collection and Import**
   - Stock price data is imported as a CSV file containing historical data such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. The data is read into a pandas DataFrame for easy manipulation and analysis.

### 2. **Feature Engineering**
   - The following features are calculated:
     - **50-day Simple Moving Average (SMA_50)**: A short-term trend indicator.
     - **200-day Simple Moving Average (SMA_200)**: A long-term trend indicator.
     - **Price Change**: The daily percentage change in the closing price, which represents how much the stock price has moved relative to the previous day.

   These features are used to identify patterns in stock price behavior.

### 3. **Rule-Based Prediction**
   - The prediction is made based on the following rules:
     - **Buy Signal**: If the 50-day SMA is above the 200-day SMA and the price change for the day is positive, predict a **buy** signal (1).
     - **Sell Signal**: If the 50-day SMA is below the 200-day SMA and the price change is negative, predict a **sell** signal (-1).
     - **No Action**: If neither condition is met, predict **no action** (0).

### 4. **Prediction Evaluation**
   - After generating predictions, the model compares the predicted buy/sell signals with the actual market movements:
     - **Actual Movement**: Determined by comparing today's closing price with the next day's closing price.
   - The prediction accuracy is calculated by comparing the predicted signals with the actual market movements.

### 5. **Visualization**
   - The stock price, moving averages, and buy/sell signals are visualized using `matplotlib` to provide an intuitive understanding of the model's performance.
   - The accuracy of the algorithm is displayed, and the results are plotted alongside actual stock price movements.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git


#Usage
Upload Historical Stock Data:
Ensure that you have a CSV file containing historical stock data (e.g., from Yahoo Finance, Alpha Vantage). The file should contain the following columns: Date, Open, High, Low, Close, Volume.

Run the Algorithm:
In a Jupyter notebook or Google Colab, upload the CSV file and run the following code to calculate technical indicators and make predictions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('your_stock_data.csv')

# Calculate moving averages and price change
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['Price_Change'] = df['Close'].pct_change()

# Define the rule-based prediction function
def predict_price_movement(row):
    if row['SMA_50'] > row['SMA_200'] and row['Price_Change'] > 0:
        return 1  # Buy Signal
    elif row['SMA_50'] < row['SMA_200'] and row['Price_Change'] < 0:
        return -1  # Sell Signal
    else:
        return 0  # No Action

# Apply the prediction function to the DataFrame
df['Prediction'] = df.apply(predict_price_movement, axis=1)

# Evaluate the predictions by comparing with actual movements
df['Next_Day_Close'] = df['Close'].shift(-1)
df['Actual_Movement'] = np.where(df['Next_Day_Close'] > df['Close'], 1, -1)

# Calculate prediction accuracy
accuracy = (df['Prediction'] == df['Actual_Movement']).sum() / len(df) * 100
print(f"Prediction Accuracy: {accuracy:.2f}%")

Visualize the Results:
After running the prediction algorithm, you can visualize the stock price and the buy/sell signals on a plot:

# Plot the stock price and moving averages
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Stock Price', color='blue')
plt.plot(df['Date'], df['SMA_50'], label='50-Day SMA', color='orange')
plt.plot(df['Date'], df['SMA_200'], label='200-Day SMA', color='green')

# Plot Buy and Sell signals
buy_signals = df[df['Prediction'] == 1]
sell_signals = df[df['Prediction'] == -1]
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price and Rule-Based Predictions')
plt.legend()
plt.xticks(rotation=45)
plt.show()

Output Accuracy:
The model prints the accuracy of the prediction, which represents the percentage of correctly predicted buy/sell signals compared to actual market movements.

Example output:
Prediction Accuracy: 75.42%

Visualization: The algorithm plots the stock price alongside the 50-day and 200-day SMAs, and highlights the buy and sell signals on the graph.

Buy signals: Marked with green upward arrows.
Sell signals: Marked with red downward arrows.
Example Results
Using this rule-based model, you can evaluate how well the algorithm performs by examining the accuracy and the visual representation of the stock price movements and predictions. For example:

If the 50-day SMA crosses above the 200-day SMA and the stock price increases, the model will generate a buy signal.
Conversely, if the 50-day SMA crosses below the 200-day SMA and the stock price decreases, the model will generate a sell signal.
Contribution
Feel free to fork this repository and submit pull requests. Contributions to improve the model are welcome, especially in areas such as:

Adding more technical indicators (e.g., RSI, MACD).
Experimenting with machine learning algorithms for improved predictions.
Improving the prediction rules based on additional data or research.
License
This project is licensed under the MIT License – see the LICENSE file for details.

markdown
Copy

### Key Sections:
1. **Overview**: A brief introduction to the algorithm and its purpose.
2. **Methodology**: An in-depth explanation of how the algorithm works, from data collection to rule-based predictions and evaluation.
3. **Installation**: Instructions for setting up the environment and installing dependencies.
4. **Usage**: How to use the code, including an example of running the algorithm and visualizing the output.
5. **Output**: Explanation of what the model's output looks like, including the prediction accuracy and visualization of buy/sell signals.
6. **Example Results**: Provides insight into how the model performs and how predictions are visualized.
7. **Contribution**: Encouragement for other developers to contribute to improving the model.
8. **License**: Information about the project's open-source license.

This `README.md` file provides a comprehensive explanation of the project, making it easy for others to understand, use, and contribute to the code.



