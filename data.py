import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import yfinance as yf
import talib
from yfinance import Ticker



class dataGenerator():

    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def read_economic_data(self):
        df_economic = pd.read_csv('WorldEconomicData.csv')
        df_economic['Date'] = pd.to_datetime(df_economic['Date'], format='%d-%b-%y')
        return df_economic
        
    def compute_score(self,row, ticker):
        print(f"Processing date: {row['Date']}")  # Debugging statement
        headlines = " ".join(map(str, row[1:]))
        prompt_positive = f"The outlook for {ticker} stock is highly positive."
        prompt_negative = f"The outlook for {ticker} stock is highly negative."

        # Get embeddings for headlines
        input_ids = self.tokenizer(headlines, return_tensors="pt", max_length=512, truncation=True)["input_ids"]

        with torch.no_grad():
            embeddings_headlines = self.model(input_ids).last_hidden_state.mean(dim=1)
        
        # Get embeddings for positive prompt + headlines
        input_ids_positive = self.tokenizer(headlines + prompt_positive, return_tensors="pt", max_length=512, truncation=True)["input_ids"]

        with torch.no_grad():
            embeddings_positive = self.model(input_ids_positive).last_hidden_state.mean(dim=1)
        
        # Get embeddings for negative prompt + headlines
        input_ids_negative = self.tokenizer(headlines + prompt_negative, return_tensors="pt", max_length=512, truncation=True)["input_ids"]
        with torch.no_grad():
            embeddings_negative = self.model(input_ids_negative).last_hidden_state.mean(dim=1)
        
        # Compute cosine similarity
        sim_positive = cosine_similarity(embeddings_headlines.cpu().numpy(), embeddings_positive.cpu().numpy())
        sim_negative = cosine_similarity(embeddings_headlines.cpu().numpy(), embeddings_negative.cpu().numpy())
        
        # Combine scores and normalize to [-1, 1]
        score = sim_positive - sim_negative
        return score[0][0]
    
    def fetch_financial_data(self, ticker, start_date, end_date):
        buffer_days = 50  
        adjusted_start_date = start_date - pd.Timedelta(days=buffer_days)
        
        # Fetch data with yfinance using the adjusted start date
        df_financial = yf.download(ticker, start=adjusted_start_date, end=end_date)
        return df_financial
    
    def add_technical_indicators(self, df):
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['Upper Band'], df['Middle Band'], df['Lower Band'] = talib.BBANDS(df['Close'], timeperiod=20)
        
        return df
    
    def fetch_fundamental_data(self, ticker):
        stock = Ticker(ticker)
        fundamentals = stock.info
        trailing_pe = fundamentals.get("trailingPE", None)
        forward_pe = fundamentals.get("forwardPE", None)
        dividend_yield = fundamentals.get("dividendYield", None)
        
        return trailing_pe, forward_pe, dividend_yield
    
    def handle_missing_data(self, df):
        # Forward fill NaN values in the 'Volume' column
        df['Volume'].fillna(method='ffill', inplace=True)
        
        # New Columns: Technical Indicators
        columns_technical = ['RSI', 'MACD', 'MACD Signal', 'Upper Band', 'Middle Band', 'Lower Band']
        
        # New Columns: Fundamental Data
        columns_fundamental = ['Trailing PE', 'Forward PE', 'Dividend Yield']
        
        # Existing Columns: Price Data
        columns_price = ['Open', 'High', 'Low', 'Close', 'Sentiment Score']

        # Handle Dividend Yield by setting NaN values to 0
        df['Dividend Yield'].fillna(0, inplace=True)

        # For other columns, fill NaN with the last available value
        for col in columns_technical + columns_price + columns_fundamental:
            if col != 'Dividend Yield':
                df[col].fillna(method='ffill', inplace=True)
        
        return df



        
    def score(self, ticker):
        df = pd.read_csv('WorldNewsData.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

        df_economic = self.read_economic_data()

        # Find the earliest ENDING date between the two CSVs
        min_end_date = min(df['Date'].max(), df_economic['Date'].max())

        # Filter the dataframes to remove dates that go beyond the earliest ENDING date
        df = df[df['Date'] <= min_end_date]
        df_economic = df_economic[df_economic['Date'] <= min_end_date]

        output_path = f"data/Data_{ticker}.csv"

        if os.path.exists(output_path):
            # Read the existing DataFrame
            existing_df = pd.read_csv(output_path)
            last_date_existing = pd.to_datetime(existing_df['Date'].iloc[-1])

            # Filter the main DataFrame to only process entries after the last_date_existing
            df = df[df['Date'] > last_date_existing]

        # If there are new entries in df after filtering, compute sentiment scores
        if not df.empty:
            df['Sentiment Score'] = df.apply(lambda row: self.compute_score(row, ticker), axis=1)

            start_date = df['Date'].iloc[0]
            end_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)

            df_financial = self.fetch_financial_data(ticker, start_date, end_date)

            # Add technical indicators
            df_financial = self.add_technical_indicators(df_financial)

            # Fetch fundamental data
            trailing_pe, forward_pe, dividend_yield = self.fetch_fundamental_data(ticker)
            df_financial['Trailing PE'] = [trailing_pe] * df_financial.shape[0]
            df_financial['Forward PE'] = [forward_pe] * df_financial.shape[0]
            df_financial['Dividend Yield'] = [dividend_yield] * df_financial.shape[0]

            # Merge sentiment data with financial data and economic data
            df_combined = pd.merge(df, df_financial, left_on="Date", right_index=True, how="left")
            df_combined = pd.merge(df_combined, df_economic, on="Date", how="left")

            # Handle missing data
            df_combined = self.handle_missing_data(df_combined)

            # Add the 'next day high' column
            df_combined["Next Day High"] = df_combined["High"].shift(-1)

            # Define the columns_output list
            columns_output = [
                'Date', 'Sentiment Score', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'RSI', 'MACD', 'MACD Signal', 'Upper Band', 'Middle Band', 'Lower Band',
                'Trailing PE', 'Forward PE', 'Dividend Yield', 'GDP', 'UNRATE', 'T5YIE', 
                'FEDFUNDS', 'CPIAUCSL', 'PPIACO', 'BOPGSTB', 'RSXFS', 'INDPRO', 'SP500', 
                'EUR=X', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 
                'XLV', 'XLY', 'Next Day High'
            ]
            output_df = df_combined[columns_output].copy()

            # Convert 'Date' column to 'YYYY-MM-DD' format without time details
            output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')

            # Set the 'Next_Day_High' for the last row to NaN
            output_df.at[output_df.index[-1], 'Next Day High'] = float('NaN')

            # Append to existing or save new
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, output_df], ignore_index=True)
                combined_df.to_csv(output_path, index=False)
            else:
                output_df.to_csv(output_path, index=False)

        else:
            print(f"No new entries found for {ticker}. The data is up-to-date.")


sc = dataGenerator()
sc.score("LMT")

