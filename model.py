import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

class StockPredictor:

    def __init__(self, ticker):
        self.ticker = ticker
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = f"data/Model_{ticker}.h5"
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = None

    def load_data(self):
        data_path = f"data/Data_{self.ticker}.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} does not exist.")
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to datetime and take only the date part
        return df.copy(deep=True)

    def preprocess_data(self, df):
        dataset = df['Next_Day_High'].values
        dataset = dataset.astype('float32')
        dataset = np.reshape(dataset, (-1, 1))
        dataset = self.scaler.transform(dataset)

        X, Y = [], []
        look_back = 20
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def train(self):
        df = self.load_data()
        train_size = int(len(df) * 0.67)
        train = df.iloc[0:train_size]
        test = df.iloc[train_size:]

        # Fit scaler on training data only
        self.scaler.fit(train[['Next_Day_High']].values)

        X_train, Y_train = self.preprocess_data(train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Model architecture
        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)  # Using batch size of 32
        self.model.save(self.model_path)

    def predict(self, recent_data):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model before predicting.")

        recent_data_copy = recent_data.copy(deep=True)
        X, _ = self.preprocess_data(recent_data_copy)
        if X.shape[0] == 0:
            raise ValueError("Not enough records in recent_data to make a prediction.")

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predicted = self.model.predict(X)
        return self.scaler.inverse_transform(predicted)

    def load_recent_data(self, end_date_str):
        df = self.load_data()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()  # Take only the date part
        start_date = end_date - timedelta(days=30)

        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        recent_data = df[mask]
        return recent_data


# Usage:
ticker = 'GOOG'
prd = StockPredictor(ticker)
prd.train()

today_str = datetime.today().strftime('%Y-%m-%d')
recent_data = prd.load_recent_data('2018-07-13')
predictions = prd.predict(recent_data)
last_date = recent_data.iloc[-1]['Date'].strftime('%Y-%m-%d')  # Format date to yyyy-mm-dd before printing
print(f"Predicted next-day-high for {ticker} on {last_date} is: {predictions[-1][0]}")