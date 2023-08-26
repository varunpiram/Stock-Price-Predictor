import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import yfinance as yf
import talib
from yfinance import Ticker


# Data generator class - generates and updates data files storing data for specific tickers
class dataGenerator:

    # Initializes tokenizer and model
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Reads economic data from csv file
    def read_economic_data(self):
        df_economic = pd.read_csv('WorldEconomicData.csv')
        # Datetime conversion
        df_economic['Date'] = pd.to_datetime(df_economic['Date'], format='%d-%b-%y')
        return df_economic

    # Computes a sentiment score on how a day of news headlines relates to a specific stock ticker
    # Does this by getting DistilBERT embeds for headlines and for negatively and positively modified
    # versions of the headlines, and computing differences in cosine similarity
    def compute_score(self,row, ticker, sentimentScore):

        if sentimentScore:
            print(f"Processing date: {row['Date']} for ticker: {ticker}") # Keeps track of progress

            # Combines headlines into str
            headlines = " ".join(map(str, row[1:]))

            # Sets positive and negative prompts
            prompt_positive = f"The outlook for {ticker} stock is highly positive."
            prompt_negative = f"The outlook for {ticker} stock is highly negative."

            # Gets embeddings for headlines
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
            
            # Combine scores
            score = sim_positive - sim_negative
            return score[0][0]
        else:
            return -1
    
    # Fetches financial data from yfinance using a buffer to allow for indicator calculations
    def fetch_financial_data(self, ticker, start_date, end_date):

        # Sets buffer day count and adjusted start date
        buffer_days = 50  
        adjusted_start_date = start_date - pd.Timedelta(days=buffer_days)
        
        # Fetches data from yfinance
        try:
            df_financial = yf.download(ticker, start=adjusted_start_date, end=end_date)
        except:
            raise Exception("Couldn't fetch yfinance data")
        return df_financial
    
    # Adds technical indicators to financial data
    def add_technical_indicators(self, df):
        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        # MACD and MACD Signal w/ appropriate parameters
        df['MACD'], df['MACD Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        return df
    
    # Fetches fundamental data from yfinance
    def fetch_fundamental_data(self, ticker):
        stock = Ticker(ticker)
        fundamentals = stock.info
        # Forward PE
        forward_pe = fundamentals.get("forwardPE", None)
        return forward_pe
    

    # Forward fills NaN values to allow for further calculations/model training
    def handle_missing_data(self, df):

        # Forward fill NaN values in the 'Volume' column
        df['Volume'].fillna(method='ffill', inplace=True)
        
        # Technical indicators
        columns_technical = ['RSI', 'MACD', 'MACD Signal']
        
        # Fundamental data
        columns_fundamental = ['Forward PE']
        
        # Price data
        columns_price = ['Open', 'High', 'Low', 'Close', 'Sentiment Score']

        # Forward fill
        for col in columns_technical + columns_price + columns_fundamental:
                df[col].fillna(method='ffill', inplace=True)
        
        return df



    # Main method which updates/creates data files for specific tickers
    def score(self, ticker, sentimentScore=True):

        # Reads the news data to get dates
        df = pd.read_csv('WorldNewsData.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

        # Reads the financial data to get dates
        df_economic = self.read_economic_data()

        # Finds the earliest ending date between the two CSVs
        min_end_date = min(df['Date'].max(), df_economic['Date'].max())

        # Filters the dataframes to remove dates that go beyond the earliest ending date
        df = df[df['Date'] <= min_end_date]
        df_economic = df_economic[df_economic['Date'] <= min_end_date]
        output_path = f"data/Data_{ticker}.csv"

        # If data (csv) for ticker exists, then reads it and grabs the last entry's date (and filters
        # main dataframe)
        if os.path.exists(output_path):

            # Read the existing data
            existing_df = pd.read_csv(output_path)

            # Drops rows with NaN values in the 'Next Day High' column (so data  is regenerated)
            existing_df.dropna(subset=['Next Day High'], inplace=True)
            existing_df.to_csv(output_path, index=False)
            existing_df = pd.read_csv(output_path)

            #existing_df.dropna(inplace=True)
            last_date_existing = pd.to_datetime(existing_df['Date'].iloc[-1])

            # Filter the main DataFrame to only process entries after the last entry
            df = df[df['Date'] > last_date_existing]

        # If there are new entries, then compute the sentiment score for each entry and grab
        # necessary data and add it to the existing data (csv)
        if not df.empty:
            # Computes sentiment score
            df['Sentiment Score'] = df.apply(lambda row: self.compute_score(row, ticker, sentimentScore), axis=1)

            # Get the appropriate dates
            start_date = df['Date'].iloc[0]
            end_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)

            # Fetch financial data
            df_financial = self.fetch_financial_data(ticker, start_date, end_date)

            # Add technical indicators
            df_financial = self.add_technical_indicators(df_financial)

            # Fetch fundamental data
            forward_pe = self.fetch_fundamental_data(ticker)
            df_financial['Forward PE'] = [forward_pe] * df_financial.shape[0]
  
            # Merge sentiment data with financial data and economic data
            df_combined = pd.merge(df, df_financial, left_on="Date", right_index=True, how="left")
            df_combined = pd.merge(df_combined, df_economic, on="Date", how="left")

            # Handle missing data
            df_combined = self.handle_missing_data(df_combined)

            # Add the 'next day high' column
            df_combined["Next Day High"] = df_combined["High"].shift(-1)

            # Define conversion of sectors to ETF tickers
            SECTOR_TO_ETF = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Basic Materials': 'XLB'
            }

            # Get the sector of the ticker
            try:
                sector = SECTOR_TO_ETF[yf.Ticker(ticker).info.get('sector')]
            except:
                print("Ticker unsupported")
                raise Exception("Ticker unsupported")



            # Defines the columns to output (with specific sector ETF)
            columns_output = [
                'Date', 'Sentiment Score', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'RSI', 'MACD', 'MACD Signal',
                'Forward PE', 'T5YIE', 
                'FEDFUNDS', 'SP500', 
                'EUR=X', sector, 'Next Day High'
            ]

            # Create the output DataFrame
            output_df = df_combined[columns_output].copy()

            # Convert 'Date' column to 'YYYY-MM-DD' format without time details
            output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')

            #Set the 'Next_Day_High' for the last row to NaN
            output_df.at[output_df.index[-1], 'Next Day High'] = float('NaN')

            # Append to existing or save new
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, output_df], ignore_index=True)
                combined_df.to_csv(output_path, index=False)
            else:
                output_df.to_csv(output_path, index=False)

        # For cases of no new entries
        else:
            print(f"No new entries found for {ticker}. The data is up-to-date.")


