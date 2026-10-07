Here is a complete `README.md` file designed for your stock price prediction project based on the PyTorch LSTM notebook provided.

---

# AAPL Stock Price Prediction using PyTorch & LSTM

An end-to-end deep learning project built in Python using **PyTorch**, **yFinance**, and **Scikit-Learn**. This project downloads historical time-series data for Apple Inc. (`AAPL`), processes it using a sliding window approach, and trains a 2-layer Long Short-Term Memory (LSTM) recurrent neural network to predict future stock closing prices.

---

## 📌 Features

* **Automated Data Retrieval:** Fetches historical daily stock data directly from Yahoo Finance via `yfinance`.


* **Time-Series Preprocessing:** Features feature scaling using `MinMaxScaler` and custom sliding window sequence generation.


* **Chronological Split:** Implements a time-series train/test split without shuffling to prevent data leakage.


* **Custom PyTorch Pipeline:** Features custom PyTorch `Dataset` and `DataLoader` classes for batch processing.


* **LSTM Neural Network:** A 2-layer LSTM model with built-in Dropout to reduce overfitting and a fully connected linear layer for price regression.


* **Dynamic Learning Rate:** Uses PyTorch’s `ReduceLROnPlateau` scheduler to adjust the learning rate during training based on validation loss.


* **Model Evaluation & Visualization:** Evaluates predictions on original dollar values using **MAE**, **RMSE**, and **$R^2$ Score**, complete with Matplotlib visualization plots.



---

## 🚀 Project Pipeline

```
Raw Stock Data (yFinance)
       │
       ▼
MinMax Scaling [0, 1]
       │
       ▼
Sliding Window Sequences (seq_length = 60)
       │
       ▼
PyTorch Dataset & DataLoader
       │
       ▼
2-Layer LSTM Model (Hidden Dim: 64, Dropout: 0.2)
       │
       ▼
Train Loop (MSE Loss + Adam Optimizer + LR Scheduler)
       │
       ▼
Inverse Scaling & Metric Evaluation (MAE, RMSE, R²)

```

---

## 🛠️ Installation & Setup

### 1. Prerequisites

Ensure you have Python 3.8+ installed along with GPU support (optional, but recommended for PyTorch).

### 2. Install Required Packages

Install the required libraries using `pip`:

```bash
pip install torch yfinance numpy pandas matplotlib scikit-learn

```

---

## 💻 Usage

Run the Jupyter Notebook or Google Colab file sequentially:

1. **Imports & Setup:** Verifies CUDA availability (`cuda` or `cpu`).


2. **Data Download & Preprocessing:** Downloads AAPL daily stock prices from `2015-01-01` to `2024-01-01`, scales closing prices, creates 60-day sequence windows, and sets up data loaders with batch size `32`.


3. **Model Architecture:** Defines the `StockLSTM` class:


* `input_dim`: 1
* `hidden_dim`: 64
* `num_layers`: 2
* `dropout`: 0.2
* `output_dim`: 1


4. **Training:** Trains the network for 40 epochs using `nn.MSELoss()` and the `Adam` optimizer with a learning rate of `0.001`.


5. **Evaluation:** Inverse scales predictions back to actual dollar amounts, prints error metrics, and plots loss curves alongside Real vs. Predicted prices.



---

## 📊 Model Architecture

```python
StockLSTM(
  (lstm): LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=64, out_features=1, bias=True)
)

```

---

## 📈 Evaluation Metrics & Outputs

The model evaluates testing performance using real-dollar values after applying inverse feature scaling:

* **MAE (Mean Absolute Error):** Measures average absolute deviation in dollars.


* **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors.


* **$R^2$ Score:** Evaluates goodness-of-fit for the predicted trajectory against ground truth.



### Plots Generated

1. **Training & Testing Loss Curve:** Plots MSE loss across all 40 epochs.


2. **AAPL Price Prediction:** Overlays actual stock prices vs. predicted stock prices on unseen test data.



---
