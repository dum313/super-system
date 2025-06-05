import csv
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

class TradingAgent:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.model = LogisticRegression(learning_rate, n_iters)

    def load_prices(self, filename):
        prices = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prices.append(float(row['Close']))
        return np.array(prices)

    def _sma(self, data, window):
        if len(data) < window:
            return np.array([])
        ret = np.convolve(data, np.ones(window)/window, mode='valid')
        return ret

    def _build_features(self, prices):
        returns = np.diff(prices) / prices[:-1]
        sma_short = self._sma(prices[:-1], 5)
        sma_long = self._sma(prices[:-1], 20)
        length = min(len(returns), len(sma_short), len(sma_long))
        returns = returns[-length:]
        sma_short = sma_short[-length:]
        sma_long = sma_long[-length:]
        sma_diff = sma_short - sma_long
        X = np.column_stack([returns, sma_diff])
        return X

    def _build_labels(self, prices):
        returns = np.diff(prices) / prices[:-1]
        return np.where(returns > 0, 1, 0)

    def train(self, csv_file):
        prices = self.load_prices(csv_file)
        X = self._build_features(prices)
        y = self._build_labels(prices)[-len(X):]
        self.model.fit(X, y)

    def predict_action(self, recent_prices):
        X = self._build_features(recent_prices)
        if len(X) == 0:
            return 'hold'
        pred = self.model.predict(X[-1].reshape(1, -1))[0]
        return 'buy' if pred == 1 else 'sell'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple AI trading agent")
    parser.add_argument('--data', required=True, help='CSV file with Close prices')
    parser.add_argument('--lookback', type=int, default=40, help='Number of recent prices to use')
    args = parser.parse_args()

    agent = TradingAgent()
    agent.train(args.data)
    prices = agent.load_prices(args.data)
    recent = prices[-args.lookback:]
    action = agent.predict_action(recent)
    print('Next action:', action)
