# Stock Price Predictor

Work in progress

WorldNewsData.csv from https://www.kaggle.com/datasets/suruchiarora/top-25-world-news-2018-2023?resource=download


# Stock Price Predictor:

## Description
### Overview:
This project allows users to predict the next day's highs for a specific stock, given sentiment-scored news data and historic financial data. It does this by using a Recurrent Neural Network with Long Short-Term Memory cells for price prediction. Sentiment analysis is done by comparing embeddings from the DistilBERT LLM through cosine similarity, and historic financial and economic data is gathered through the yfinance API and the FRED API. 

Note: Some news data has been collected from this Kaggle dataset: https://www.kaggle.com/datasets/suruchiarora/top-25-world-news-2018-2023?resource=download. Default data goes back to mid-June 2018 - some data is unavailable prior to this.
### Structure:
The project allows for users to select specific stocks, load and save models for prediction or further training, and view the model's performance via through visualizing test data and through backtesting over a custom range. Additionally, the project allows for some customization through selection of
epoch count, batch size, and lookback period (although some default values are recommended).

The project will first update aggregated news and economic data by inferring headlines from URLs gathered by GDELT for dates that have not yet been processed. It will then gather relevant financial data from yfinance and economic data from FRED, and save it accordingly to CSV files meant for storing data relevant to all stock tickers.

It will then allow users the option to update or create data specific to one stock, which is done by gathering appropriate financial information via yfinance and then scoring sentiment by comparing cosine similarity of news headlines to positively and negatively modified versions via DistilBERT. This data is then combined and saved to a CSV file specific to the stock ticker, located in the `data` directory.

The project will then allow users to train a model on the data, which is done by first loading the data from the CSV file, then splitting it into training and testing data. The model is then trained on the training data, and the testing data is used to evaluate the model's performance. The model is then saved to a csv file specific to the stock ticker.

Next, the project lets users load, create, or train models for a specific stock, which are used to predict next day highs given historic data. The models are LSTMs implemented via TensorFlow's Keras, and are trained on the aforementioned, ticker-specific data. The user can select custom hyperparameters - epoch count, lookback, and batch size - although default values are recommended. Models are stored in the `data` directory, and users may further train them or directly load them for use.

Finally, the project allows users to predict next day highs for a specific stock using the above models and data. The project allows the user to predict next day highs given a date, to display the model's performance over unseen test data, and to backtest the model over a custom range of dates.

Note: Data used for prediction must be stored within the stock's data file, however, not all of this data is used for training.
### Performance:
TBU
## Usage:
### Use:
This project can be run locally through a simple command-line interface. To run this project, simply follow setup instructions and run `app.py`.

For testing purposes, users can run an instance of the app in code via the `.run(scoreToggle=False)`method if they choose to disable sentiment scoring (as this can be very time consuming over large time periods).
### Setup:
Clone this repository, and install necessary libraries via:

`pip install -r requirements.txt`

This project requires a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html), so store this in a .env file
in the project root directory like so:
```
FRED_KEY='[key]'
```

If you would like to use the provided sample data, then simply move the files within the `sample`
directory to the `data` directory.

