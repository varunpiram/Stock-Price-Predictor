import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas_market_calendars as mcal


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
        # Gets NYSE calendar
        self.cal = mcal.get_calendar('NYSE')


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

        # If the date is not in the existing dataframe, use raw CSV data
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
    
    # Checks if a day is a NYSE trading day
    def check_trading(self, date):
        return not self.cal.schedule(start_date=date, end_date=date).empty
    

    # Backtests the model's trading strategy against a simple buy-and-hold benchmark strategy
    def backtest(self, start_date, end_date):

        # Check if the start and end dates are present in the dataset.
        if start_date not in self.data["Date"].values or end_date not in self.data["Date"].values:
            print("Either start date or end date not found in dataset!")
            return

        # Calculate the initial portfolio value based on the opening price of the start_date.
        initial_value = self.data.loc[self.data["Date"] == start_date, "Open"].values[0]

        # Lists to store portfolio values over time for both strategies: buy-and-hold (benchmark) and model.
        benchmark_values = [initial_value]
        model_values = [initial_value]

        # Counter to keep track of profits or losses for the model strategy.
        model_profit_count = 0.0

        # Define a date range starting the day after the start_date and ending on the end_date.
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')[1:]

        # List to store the dates for which the model strategy made a trade.
        dates_out = []
        dates_out.append(start_date) # Include starting date

        # Iterate through each date in the date range.
        for date in date_range:
            # Check if the current date is a trading day
            if self.check_trading(date):
                # Format date
                date_str = date.strftime('%Y-%m-%d')
                # Get past day's date
                prev_date_str = (date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

                # Use the model to predict the current day's high price based on the previous trading day's data
                predicted_high = self.predict_given_date(prev_date_str)

                # Retrieve the open, high, and close prices for the current day 
                open_price = self.data.loc[self.data["Date"] == date_str, "Open"].values[0]
                high_price = self.data.loc[self.data["Date"] == date_str, "High"].values[0]
                close_price = self.data.loc[self.data["Date"] == date_str, "Close"].values[0]

                # Add benchmark return to benchmark_values list
                benchmark_values.append(close_price)

                # If the model's predicted high is greater than the opening price, buy
                if predicted_high > open_price:
                    # If the actual high of the day surpasses the predicted high, sell at predicted high
                    if predicted_high < high_price:
                        model_profit_count += (predicted_high - open_price)
                    else:
                        # If the predicted high isn't reached, sell at close
                        model_profit_count += (close_price - open_price)

                # Append the model's updated portfolio value to the model_values list
                model_values.append(initial_value + model_profit_count)

                dates_out.append(date.strftime('%Y-%m-%d'))

        # Calculate the percentage returns for both strategies
        base_profit = (benchmark_values[-1] - initial_value) * 100 / initial_value
        model_profit = (model_values[-1] - initial_value) * 100 / initial_value

        # Plot the portfolio values over time for both strategies
        plt.figure(figsize=(14, 7))
        print(len(dates_out), len(benchmark_values), len(model_values))

        plt.plot(dates_out, benchmark_values, label='Benchmark Returns', color='blue')
        plt.plot(dates_out, model_values, label='Model Returns', color='red')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title('Model vs Benchmark Returns Over Time')



        # Set the x-axis major formatter
        ax = plt.gca()  # Get the current axes
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=8))  # Set the tick locator to every 8 days

        # Rotate x labels for better visibility
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


        plt.tight_layout()  # Adjust layout for better visibility
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print the overall percentage returns.
        print(f"Benchmark Returns: {base_profit:.2f}%")
        print(f"Model Returns: {model_profit:.2f}%")



#mod = StockPredictor("XOM")
#mod.train_model()
#mod.predict_test_data()
#mod.backtest("2023-03-01", "2023-08-01")
