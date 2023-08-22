import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


# Load the dataset
ticker = "AAPL"
data = pd.read_csv(f"data/Data_{ticker}.csv")
data = data.dropna()

# Feature selection
features = data.drop(["Date", "Next Day High"], axis=1)
targets = data["Next Day High"]

# Scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
target_scaler = MinMaxScaler()
targets_scaled = target_scaler.fit_transform(targets.values.reshape(-1, 1))

def create_dataset(X, y, lookback=1):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        v = X.iloc[i:(i + lookback)].values
        Xs.append(v)
        ys.append(y.iloc[i + lookback])
    return np.array(Xs), np.array(ys)

# Set your desired lookback here
lookback = 1

X, y = create_dataset(pd.DataFrame(features_scaled), pd.DataFrame(targets_scaled), lookback)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)

open_prices = data["Open"].values[-len(X_test):]
close_prices = data["Close"].values[-len(X_test):]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.001), loss="mean_squared_error")

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, shuffle=False)

# Predictions
y_pred = model.predict(X_test)

# Inverse transform the scaled data
y_pred_original = target_scaler.inverse_transform(y_pred)
y_test_original = target_scaler.inverse_transform(y_test)

# Extracting the last 10 dates from the test set
last_10_dates = data["Date"].values[-len(y_test):][-10:]

# Printing the last 10 actual and predicted highs along with their dates
print("\nLast 10 Predictions:")
for date, actual, pred in zip(last_10_dates, y_test_original[-10:], y_pred_original[-10:]):
    print(f"Date: {date} | Actual High: {actual[0]:.2f} | Predicted High: {pred[0]:.2f}")

# Plotting the results
plt.figure(figsize=(14, 6))
plt.plot(y_test_original, label="Actual Highs", color="blue")
plt.plot(y_pred_original, label="Predicted Highs", color="red")

plt.plot(open_prices, label="Open Prices", color="green", linewidth=0.5)
plt.plot(close_prices, label="Close Prices", color="magenta", linewidth=0.5)


plt.title("Comparison of Actual and Predicted Next Day High Prices")
plt.legend()
plt.show()
