"""Minimalist AI trading agent with optional GUI.

This module implements a very small logistic regression model for
trading on historical price data. It can read prices from a CSV file or
fetch them via the CoinGecko API. A simple Tkinter interface is included
for users who prefer a graphical front end. The code is heavily
commented to aid understanding and is designed for low-powered
computers.
"""

import csv
import json
from urllib.request import urlopen
import numpy as np

# Tkinter is only imported when running the GUI to keep the
# command-line usage lightweight.

class LogisticRegression:
    """Extremely small implementation of logistic regression."""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        # learning rate controls how much we update weights at each step
        self.lr = learning_rate
        # number of gradient descent iterations
        self.n_iters = n_iters
        # model parameters will be set during training
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for buy and 0 for sell."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return np.where(y_pred >= 0.5, 1, 0)


class TradingAgent:
    """Helper class that builds features from prices and trains the model."""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        self.model = LogisticRegression(learning_rate, n_iters)

    # ------------------------------------------------------------------
    # Data loading utilities
    # ------------------------------------------------------------------
    def load_crypto_prices(self, coin_id: str, vs_currency: str = "usd", days: int = 30) -> np.ndarray:
        """Fetch close prices from the CoinGecko API."""
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?"
            f"vs_currency={vs_currency}&days={days}"
        )
        with urlopen(url) as resp:
            data = json.load(resp)
        prices = [p[1] for p in data.get("prices", [])]
        return np.array(prices, dtype=float)

    def load_prices(self, filename: str) -> np.ndarray:
        """Load closing prices from a CSV file."""
        prices = []
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prices.append(float(row["Close"]))
        return np.array(prices)

    # ------------------------------------------------------------------
    # Training and feature construction
    # ------------------------------------------------------------------
    def train_from_prices(self, prices: np.ndarray) -> None:
        """Prepare features and train the model."""
        X = self._build_features(prices)
        y = self._build_labels(prices)[-len(X):]
        self.model.fit(X, y)

    def _sma(self, data: np.ndarray, window: int) -> np.ndarray:
        """Simple moving average helper."""
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def _build_features(self, prices: np.ndarray) -> np.ndarray:
        """Create feature matrix from raw prices."""
        returns = np.diff(prices) / prices[:-1]
        sma_short = self._sma(prices[:-1], 5)
        sma_long = self._sma(prices[:-1], 20)
        length = min(len(returns), len(sma_short), len(sma_long))
        returns = returns[-length:]
        sma_short = sma_short[-length:]
        sma_long = sma_long[-length:]
        sma_diff = sma_short - sma_long
        return np.column_stack([returns, sma_diff])

    def _build_labels(self, prices: np.ndarray) -> np.ndarray:
        """Generate 1 if price increased, otherwise 0."""
        returns = np.diff(prices) / prices[:-1]
        return np.where(returns > 0, 1, 0)

    def predict_action(self, recent_prices: np.ndarray) -> str:
        """Predict the next action from the most recent prices."""
        X = self._build_features(recent_prices)
        if len(X) == 0:
            return "hold"
        pred = self.model.predict(X[-1].reshape(1, -1))[0]
        return "buy" if pred == 1 else "sell"


# ----------------------------------------------------------------------
# Optional Tkinter-based GUI
# ----------------------------------------------------------------------

def run_gui() -> None:
    """Start a very small graphical interface for the trader."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("AI Trader")

    # Variables bound to the UI widgets
    mode = tk.StringVar(value="file")  # choose file or coin
    file_path = tk.StringVar()
    coin_id = tk.StringVar(value="bitcoin")
    days = tk.StringVar(value="30")
    lookback = tk.StringVar(value="40")

    # UI helper callbacks -------------------------------------------------
    def browse_file() -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            file_path.set(path)

    def run_trader() -> None:
        agent = TradingAgent()
        try:
            if mode.get() == "file":
                if not file_path.get():
                    messagebox.showerror("Error", "CSV file path required")
                    return
                prices = agent.load_prices(file_path.get())
            else:
                prices = agent.load_crypto_prices(coin_id.get(), days=int(days.get()))
            agent.train_from_prices(prices)
            recent = prices[-int(lookback.get()):]
            action = agent.predict_action(recent)
            messagebox.showinfo("Result", f"Next action: {action}")
        except Exception as exc:  # show any errors to the user
            messagebox.showerror("Error", str(exc))

    # Layout --------------------------------------------------------------
    row = 0
    tk.Radiobutton(root, text="Use CSV file", variable=mode, value="file").grid(row=row, column=0, sticky="w")
    row += 1
    tk.Entry(root, textvariable=file_path, width=40).grid(row=row, column=0)
    tk.Button(root, text="Browse", command=browse_file).grid(row=row, column=1)

    row += 1
    tk.Radiobutton(root, text="Fetch from CoinGecko", variable=mode, value="coin").grid(row=row, column=0, sticky="w")
    row += 1
    tk.Label(root, text="Coin id:").grid(row=row, column=0, sticky="e")
    tk.Entry(root, textvariable=coin_id).grid(row=row, column=1)

    row += 1
    tk.Label(root, text="Days:").grid(row=row, column=0, sticky="e")
    tk.Entry(root, textvariable=days).grid(row=row, column=1)

    row += 1
    tk.Label(root, text="Lookback:").grid(row=row, column=0, sticky="e")
    tk.Entry(root, textvariable=lookback).grid(row=row, column=1)

    row += 1
    tk.Button(root, text="Run", command=run_trader).grid(row=row, column=0, columnspan=2, pady=10)

    root.mainloop()


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple AI trading agent")
    parser.add_argument("--data", help="CSV file with Close prices")
    parser.add_argument("--coin", help="CoinGecko coin id (e.g. bitcoin)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch when using --coin")
    parser.add_argument("--lookback", type=int, default=40, help="Number of recent prices to use")
    parser.add_argument("--gui", action="store_true", help="Launch graphical interface")
    args = parser.parse_args()

    # Start GUI if requested
    if args.gui:
        run_gui()
        raise SystemExit

    if not args.data and not args.coin:
        parser.error("Provide --data or --coin to load prices")

    agent = TradingAgent()
    if args.data:
        prices = agent.load_prices(args.data)
    else:
        prices = agent.load_crypto_prices(args.coin, days=args.days)

    agent.train_from_prices(prices)
    recent = prices[-args.lookback:]
    action = agent.predict_action(recent)
    print("Next action:", action)
