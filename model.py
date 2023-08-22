import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


class StockPredictor:

    def __init__(self, ticker):
        self.ticker = ticker
        self.data = self._load_data()
        self.features_scaled, self.targets_scaled = self._preprocess_data()
        self.model_path = f"data/Model_{ticker}.h5"
        self.model = self._get_model()
        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(self.data["Next Day High"].values.reshape(-1, 1))

    def _load_data(self):
        data = pd.read_csv(f"data/Data_{self.ticker}.csv")
        return data.dropna()

    def _preprocess_data(self):
        features = self.data.drop(["Date", "Next Day High"], axis=1)
        targets = self.data["Next Day High"]

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        target_scaler = MinMaxScaler()
        targets_scaled = target_scaler.fit_transform(targets.values.reshape(-1, 1))

        return features_scaled, targets_scaled

    def _get_model(self):
        try:
            model = load_model(self.model_path)
        except:
            model = Sequential()
            model.add(LSTM(50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(50))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")
        return model

    def create_dataset(self, lookback=1):
        Xs, ys = [], []
        for i in range(len(self.features_scaled) - lookback):
            v = self.features_scaled[i:(i + lookback)]
            Xs.append(v)
            ys.append(self.targets_scaled[i + lookback])
        return np.array(Xs), np.array(ys)

    def train_model(self, lookback=1, epochs=50, batch_size=32):
        X, y = self.create_dataset(lookback)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), shuffle=False)
        self.model.save(self.model_path)

    def predict_test_data(self, lookback=1):
        X, y = self.create_dataset(lookback)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        # Get the corresponding date range for the test set
        date_range = self.data["Date"].values[-len(y_test):]

        y_pred = self.model.predict(X_test)
        y_pred_original = self.target_scaler.inverse_transform(y_pred)
        y_test_original = self.target_scaler.inverse_transform(y_test)

        # Print the last 10 actual vs predicted values
        for i in range(1, 11):
            print(f"Date: {date_range[-i]}, Actual: {y_test_original[-i][0]}, Predicted: {y_pred_original[-i][0]}")

        plt.figure(figsize=(14, 6))
        plt.plot(date_range, y_test_original, label="Actual Highs", color="blue")
        plt.plot(date_range, y_pred_original, label="Predicted Highs", color="red")
        plt.title("Actual and Predicted Next-Day High Prices")
        plt.xticks(date_range[::5], rotation=45)  # Rotate x-axis labels for better visibility
        plt.tight_layout()  # Adjust layout for better visibility
        plt.legend()
        plt.show()


    def predict_given_date(self, date, lookback=1):
        # Check if the date exists in the dataframe
        date_rows = self.data[self.data["Date"].astype(str).str.strip() == date.strip()]

        # If the date is not in the dataframe (special edge case for the most recent date), use raw CSV data
        if date_rows.empty:
            raw_data = pd.read_csv(f"data/Data_{self.ticker}.csv")
            date_idx_raw = raw_data[raw_data["Date"].astype(str).str.strip() == date.strip()].index[0]

            # Ensure there's enough data for the lookback
            if date_idx_raw - lookback + 1 < 0:
                print("Not enough data for the given date and lookback.")
                return

            # Extract the lookback data, drop the Date and Next Day High columns
            raw_lookback_data = raw_data.iloc[date_idx_raw - lookback + 1:date_idx_raw + 1].drop(["Date", "Next Day High"], axis=1)

            # Scale the features with the previously used scaler
            features_for_prediction = self.scaler.transform(raw_lookback_data)

            X = np.array([features_for_prediction])
            
        else:
            idx = date_rows.index[0]
            
            if idx - lookback + 1 < 0:
                print("Not enough data for the given date and lookback.")
                return

            # Use preprocessed data from the main dataframe
            X = np.array([self.features_scaled[idx - lookback + 1:idx + 1]])

        # Predict the "Next Day High"
        prediction = self.model.predict(X)
        predicted_high = self.target_scaler.inverse_transform(prediction)
        
        print(f"Predicted next-day high for {self.ticker} on {date} is {str(predicted_high[0][0])}")
        return predicted_high[0][0]
    
    #TODO: Does not work
    def backtest(self, start_date, end_date):
        # Ensure the dates are present in the dataset
        if start_date not in self.data["Date"].values or end_date not in self.data["Date"].values:
            print("Either start date or end date not found in dataset!")
            return

        # Base profit calculation
        base_profit = (self.data.loc[self.data["Date"] == end_date, "Close"].values[0] /
                    self.data.loc[self.data["Date"] == start_date, "Open"].values[0]) * 100
        
        base_profit = base_profit - 100

        model_profit_count = 0.0

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')[1:]  # Start from the next day of start_date
        for date in date_range:
            print(f"Running profit on date {date}: {model_profit_count}")
            date_str = date.strftime('%Y-%m-%d')  # Convert date to string format to match the dataframe
            prev_date_str = (date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            predicted_high = self.predict_given_date(prev_date_str)  # Use previous day's data to predict

            open_price = self.data.loc[self.data["Date"] == date_str, "Open"].values[0]
            high_price = self.data.loc[self.data["Date"] == date_str, "High"].values[0]
            close_price = self.data.loc[self.data["Date"] == date_str, "Close"].values[0]

            if predicted_high > open_price:
                if predicted_high < high_price:
                    model_profit_count += (high_price - open_price)
                else:
                    model_profit_count += (close_price - open_price)

        # Calculate model profit percentage
        model_cumulative_value = self.data.loc[self.data["Date"] == start_date, "Open"].values[0] + model_profit_count
        model_profit_percentage = (model_cumulative_value / self.data.loc[self.data["Date"] == start_date, "Open"].values[0]) * 100
        model_profit = model_profit_percentage - 100

        print(f"Base Profit: {base_profit:.2f}%")
        print(f"Model Profit: {model_profit:.2f}%")




predictor = StockPredictor("AAPL")
#predictor.train_model(lookback=1, epochs=50, batch_size=32)
predictor.backtest(start_date='2018-07-01', end_date='2023-08-20')
#predictor.predict_test_data()
#predictor.predict_given_date('2023-08-21')
