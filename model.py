import numpy as np
import pandas as pd


from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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
        df = pd.read_csv(data_path)
        return df
    
    def preprocess_data(self, df):
        dataset = df['Next_Day_High'].values
        dataset = dataset.astype('float32')
        dataset = np.reshape(dataset, (-1, 1))
        dataset = self.scaler.fit_transform(dataset)
        
        # Convert to supervised learning problem
        X, Y = [], []
        look_back = 1
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    def train(self):
        df = self.load_data()
        X, Y = self.preprocess_data(df)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            
        self.model.fit(X, Y, epochs=50, batch_size=1, verbose=1)
        self.model.save(self.model_path)
                
    def predict(self, recent_data):
        X, _ = self.preprocess_data(recent_data)
        
        if X.shape[0] == 0:
            raise ValueError("Not enough records in recent_data to make a prediction.")

        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        predicted = self.model.predict(X)
        return self.scaler.inverse_transform(predicted)
    
    def load_data(self):
        data_path = f"data/Data_{self.ticker}.csv"
        df = pd.read_csv(data_path)
        return df.tail(5)  # This line ensures you only get the last 5 rows
    

# Usage:
# predictor = StockPredictor('AAPL')
# predictor.train()
# recent_data = pd.DataFrame([...])
# predictions = predictor.predict(recent_data)


prd = StockPredictor('AAPL')
prd.train()

# Now, for prediction, just load the data again and use it:
recent_data = prd.load_data()

predictions = prd.predict(recent_data)


last_date = recent_data.iloc[-1]['Date']  # Assuming the date column in your dataframe is named 'Date'
print(f"Predicted high for {last_date} is: {predictions[-1][0]}")  # us
