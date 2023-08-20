

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import yfinance as yf



class sentimentScorer():

    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
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
        # Fetch data with yfinance
        df_financial = yf.download(ticker, start=start_date, end=end_date)
        return df_financial
    
    def handle_missing_data(self, df):
        # Forward fill NaN values in the 'Volume' column
        df['Volume'].fillna(method='ffill', inplace=True)

        # For other columns, fill NaN with the last available 'Close' value
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            df[col].fillna(method='ffill', inplace=True)
        return df

        
    def score(self, ticker):
        df = pd.read_csv('WorldNewsData.csv')
        output_path = f"data/Data_{ticker}.csv"
        if os.path.exists(output_path):
            # Read the existing DataFrame
            existing_df = pd.read_csv(output_path)
            last_date_existing = pd.to_datetime(existing_df['Date'].iloc[-1])
            # Filter the main DataFrame to only process entries after the last_date_existing
            df = df[pd.to_datetime(df['Date']) > last_date_existing]

        # If there are new entries in df after filtering, compute sentiment scores
        if not df.empty:
            df['Sentiment_Score'] = df.apply(lambda row: self.compute_score(row, ticker), axis=1)
            start_date = pd.to_datetime(df['Date'].iloc[0])

            end_date = pd.to_datetime(df['Date'].iloc[-1]) + pd.Timedelta(days=1)
            
            df_financial = self.fetch_financial_data(ticker, start_date, end_date)
            
            # Merge sentiment data with financial data
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
            df_combined = pd.merge(df, df_financial, left_on="Date", right_index=True, how="left")
            df_combined = self.handle_missing_data(df_combined)
            
            # Add the 'next day high' column
            df_combined["Next_Day_High"] = df_combined["High"].shift(-1)
            

            # Prepare the new DataFrame to save
            output_df = df_combined[['Date', 'Sentiment_Score', 'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Day_High']]
            
            output_df = output_df.drop(output_df.index[-1])
            
            # Append to existing or save new
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                final_df = pd.concat([existing_df, output_df], ignore_index=True)
                final_df.to_csv(output_path, index=False)
            else:
                if not os.path.exists("data"):
                    os.makedirs("data")
                output_df.to_csv(output_path, index=False)
            
            print("Data retrieved and saved successfully.")
        else:
            print(f"No new dates found. Data_{ticker}.csv is up-to-date.")


sc = sentimentScorer()
sc.score("AAPL")

