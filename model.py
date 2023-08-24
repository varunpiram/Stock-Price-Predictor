import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


# Stock predictor class - loads data, preprocesses data, creates model, and trains model
class StockPredictor:

    # Initialize class with ticker, scalers, and pre-processed data
    def __init__(self, ticker):
        self.ticker = ticker
        # Loads data
        self.data = self._load_data()
        # Preprocesses data
        self.features_scaled, self.targets_scaled = self._preprocess_data()
        # Path for model
        self.model_path = f"data/Model_{ticker}.h5"
        # Gets the model/creates model
        self.model = self._get_model()
        # Target scaler (used for inverse transform)
        self.target_scaler = MinMaxScaler()
        # Prepares scaler
        self.target_scaler.fit(self.data["Next Day High"].values.reshape(-1, 1))

    # Loads data
    def _load_data(self):
        data = pd.read_csv(f"data/Data_{self.ticker}.csv")
        # Drops rows with missing values
        return data.dropna()

    # Preprocesses data
    def _preprocess_data(self):
        # Drops date and next day high columns
        features = self.data.drop(["Date", "Next Day High"], axis=1)
        targets = self.data["Next Day High"]

        # Scales features and targets (note this is different target scaler than initialized
        # in __init__)
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        target_scaler = MinMaxScaler()
        targets_scaled = target_scaler.fit_transform(targets.values.reshape(-1, 1))

        return features_scaled, targets_scaled

    # Gets model
    def _get_model(self):
        try:
            # Tries to load model in existing path
            model = load_model(self.model_path)
        except:
            # Makes a new model if that doesn't work, 2 layers 2 dropouts
            model = Sequential()
            model.add(LSTM(50, return_sequences=True)) # LSTM layer w/ 50 neurons
            model.add(Dropout(0.2)) # Dropout layer w/ 20% dropout
            model.add(LSTM(50)) # LSTM layer w/ 50 neurons
            model.add(Dropout(0.2)) # Dropout layer w/ 20% dropout
            model.add(Dense(1)) # Dense layer w/ 1 neuron

            # Adam optimizer with mean squared error loss
            model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error") # Compiles model

        return model

    # Creates dataset for the model
    def create_dataset(self, lookback=1):
        # Creates Xs and ys
        Xs, ys = [], []
        # Iterates through the scaled features and targets - adds to Xs and ys respectively
        for i in range(len(self.features_scaled) - lookback):
            v = self.features_scaled[i:(i + lookback)]
            Xs.append(v)
            ys.append(self.targets_scaled[i + lookback])
        return np.array(Xs), np.array(ys)

    # Trains model with default 1 lookback, 50 epochs, batch size of 32
    def train_model(self, lookback=1, epochs=50, batch_size=32):
        # Creates dataset
        X, y = self.create_dataset(lookback)
        # Splits data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        # Fits model with appropriate parameters
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), shuffle=False)
        # Saves model
        self.model.save(self.model_path)

    # Predicts test data
    def predict_test_data(self, lookback=1):
        # Creates dataset
        X, y = self.create_dataset(lookback)
        # Splits data into train and test sets
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        # Get the corresponding date range for the test set
        date_range = self.data["Date"].values[-len(y_test):]

        # Predicts test data
        y_pred = self.model.predict(X_test)

        # Inverse transforms the predicted and actual values using scaler from __init__
        y_pred_original = self.target_scaler.inverse_transform(y_pred)
        y_test_original = self.target_scaler.inverse_transform(y_test)

        # Print the last 10 actual vs predicted values
        for i in range(1, 11):
            print(f"Date: {date_range[-i]}, Actual: {y_test_original[-i][0]}, Predicted: {y_pred_original[-i][0]}")

        # Plot the actual vs predicted values and display
        plt.figure(figsize=(14, 6))
        plt.plot(date_range, y_test_original, label="Actual Highs", color="blue")
        plt.plot(date_range, y_pred_original, label="Predicted Highs", color="red")
        plt.title("Actual and Predicted Next-Day High Prices")
        plt.xticks(date_range[::5], rotation=45)  # Rotate x labels for better visibility
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

            # Scale the features with the scaler from __init__ 
            features_for_prediction = self.scaler.transform(raw_lookback_data)
            
            # Provides the features in the correct shape for the model
            X = np.array([features_for_prediction])
            
        else:
            # Gets the index of the date
            idx = date_rows.index[0]
            
            # Ensures there's enough data for the lookback
            if idx - lookback + 1 < 0:
                print("Not enough data for the given date and lookback.")
                return

            # Use preprocessed data from the main dataframe
            X = np.array([self.features_scaled[idx - lookback + 1:idx + 1]])

        # Predict the "Next Day High"
        prediction = self.model.predict(X)
        # Inverse transform the prediction using scaler from __init__
        predicted_high = self.target_scaler.inverse_transform(prediction)
        
        #print(f"Predicted next-day high for {self.ticker} on {date} is {str(predicted_high[0][0])}")
        return predicted_high[0][0]
    
    def backtest(self, start_date, end_date):

        # Ensure the dates are present in the dataset
        if start_date not in self.data["Date"].values or end_date not in self.data["Date"].values:
            print("Either start date or end date not found in dataset!")
            return

        # Base profit calculation
        base_profit = (self.data.loc[self.data["Date"] == end_date, "Close"].values[0] /
                    self.data.loc[self.data["Date"] == start_date, "Open"].values[0]) * 100
        
        # Gets percent change
        base_profit = base_profit - 100

        # Running profit counter
        model_profit_count = 0.0

        # Gets the date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')[1:]  # Start from the next day of start_date

        for date in date_range:
            
            # Convert date to string format to match the dataframe
            date_str = date.strftime('%Y-%m-%d')

            # Get the previous day's date 
            prev_date_str = (date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            # Predicts today's high using previous date's data
            predicted_high = self.predict_given_date(prev_date_str)  # Use previous day's data to predict

            # Get today's open, high and close prices
            open_price = self.data.loc[self.data["Date"] == date_str, "Open"].values[0]
            high_price = self.data.loc[self.data["Date"] == date_str, "High"].values[0]
            close_price = self.data.loc[self.data["Date"] == date_str, "Close"].values[0]
            
            # If the predicted high is greater than the open price, we buy the stock at open
            if predicted_high > open_price:

                # If we hit the predicted high, then we sell there
                if predicted_high < high_price:
                    model_profit_count += (predicted_high - open_price)
                else:
                    # If we don't hit the predicted high, we sell at close
                    model_profit_count += (close_price - open_price)

        # Calculate model profit percentage
        model_cumulative_value = self.data.loc[self.data["Date"] == start_date, "Open"].values[0] + model_profit_count
        model_profit_percentage = (model_cumulative_value / self.data.loc[self.data["Date"] == start_date, "Open"].values[0]) * 100
        model_profit = model_profit_percentage - 100

        # Print the results
        print(f"Benchmark Returns: {base_profit:.2f}%")
        print(f"Model Returns: {model_profit:.2f}%")

