import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
import requests
import re
import yfinance as yf
from fredapi import Fred
import time


Fredkey = '1353cc07f1bd42904e62f96d8f984d86'


class newsKeeper:

    def __init__(self):
        self.fred = Fred(api_key=Fredkey)
        return
    
    # Infers a headlines from a url's pathway by getting rid of dashes from the final "part"
    # Has some unintended info, but it is best to prioritize accuracy over clarity as it is more
    # efficient to have LLM ignore issues rather than filter them here and potentially lose important
    # information
    def extract_title_from_url(self, url):
        # Split by slashes and get the last segment
        parts = url.strip('/').split('/')
        title = parts[-1]

        # Remove any file extension (e.g. .html, .php)
        title = re.sub(r'\.\w+$', '', title)

        # Remove any URL parameters (after a '?')
        title = title.split('?')[0]

        # Replace hyphens/underscores with spaces
        title = title.replace('-', ' ')
        title = title.replace('_', ' ')

        # Removes hashes/alphanumerical codes
        title = re.sub(r'\b[a-fA-F0-9]{32}\b', '', title)

        return title
    
    # Fetch GDELT's news information for a given date (GDELT mentions) by downloading the csv into
    # memory and extracting the URLS
    def fetch_gd(self, date):
        formatted_date = date.strftime('%Y%m%d')
        gdelt_url = f"http://data.gdeltproject.org/gdeltv2/{formatted_date}000000.mentions.CSV.zip"
        
        response = requests.get(gdelt_url, stream=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(f"{formatted_date}000000.mentions.CSV") as f:
                df = pd.read_csv(f, sep='\t', header=None)
        return df[5].tolist()
    
    def fetch_economic_data(self, date):
        formatted_date = date.strftime('%Y-%m-%d')

        print(f"Fetching data for {formatted_date}")
        
        # Fetch your economic indicators using the FRED API and yfinance
        gdp = self.fred.get_series('GDP', start_date=formatted_date, end_date=formatted_date)
        unemployment = self.fred.get_series('UNRATE', start_date=formatted_date, end_date=formatted_date)
        inflation = self.fred.get_series('CPIAUCSL', start_date=formatted_date, end_date=formatted_date)
        interest_rate = self.fred.get_series('TB3MS', start_date=formatted_date, end_date=formatted_date)
        cpi = self.fred.get_series('CPIAUCSL', start_date=formatted_date, end_date=formatted_date)
        ppi = self.fred.get_series('PPIACO', start_date=formatted_date, end_date=formatted_date)
        balance_of_trade = self.fred.get_series('BOPGSTB', start_date=formatted_date, end_date=formatted_date)
        retail_sales = self.fred.get_series('RSXFS', start_date=formatted_date, end_date=formatted_date)
        industrial_production = self.fred.get_series('INDPRO', start_date=formatted_date, end_date=formatted_date)
        sp500 = self.fred.get_series('SP500', start_date=formatted_date, end_date=formatted_date)

        
        # Fetch USD price in euros using yfinance
        usd_price_euro_data = yf.download('EUR=X', start=formatted_date, end=formatted_date)
        usd_price_euro = usd_price_euro_data['Close'][0]

        sector_tickers = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC']
        sector_data = yf.download(sector_tickers, start=formatted_date, end=formatted_date)
        
        xlk = sector_data['Close']['XLK'][0]
        xlf = sector_data['Close']['XLF'][0]
        xlv = sector_data['Close']['XLV'][0]
        xly = sector_data['Close']['XLY'][0]
        xlp = sector_data['Close']['XLP'][0]
        xle = sector_data['Close']['XLE'][0]
        xli = sector_data['Close']['XLI'][0]
        xlb = sector_data['Close']['XLB'][0]
        xlre = sector_data['Close']['XLRE'][0]
        xlu = sector_data['Close']['XLU'][0]
        xlc = sector_data['Close']['XLC'][0]
        
        return (
            gdp, unemployment, inflation, interest_rate,
            cpi, ppi, balance_of_trade, retail_sales, industrial_production, sp500, usd_price_euro,
            xlk, xlf, xlv, xly, xlp, xle, xli, xlb, xlre, xlu, xlc
        )
    
        
    
   


    # Updates WorldNewsData.csv with headlines up until the current day by fetching GDELT information,
    # extracting the titles from the URLs, and adding them to the csv file
    def updateNews(self):
        df = pd.read_csv('WorldNewsData.csv')
        last_date = pd.to_datetime(df['Date'].iloc[-1], format='%d-%b-%y')
        next_day = last_date + timedelta(days=1)
        current_day = datetime.now().date()

        while next_day <= pd.Timestamp(current_day):
            urls = self.fetch_gd(next_day)
            
            # Filter duplicate URLS
            unique_urls = list(set(urls))
            
            # Extract titles from URLs
            titles = [self.extract_title_from_url(url) for url in unique_urls][:25]
            new_entry = [next_day.strftime('%d-%b-%y')] + titles
            df.loc[len(df)] = new_entry
            next_day += timedelta(days=1)
        
        df.to_csv('WorldNewsData.csv', index=False)

    # Update WorldEconomicData.csv with economic indicators
    def updateEconomicData(self):
        df_economic = pd.read_csv('WorldEconomicData.csv')
        last_date_economic = pd.to_datetime(df_economic['Date'].iloc[-1], format='%d-%b-%y')
        next_day_economic = last_date_economic + timedelta(days=1)
        current_day_economic = datetime.now().date()

        while next_day_economic <= pd.Timestamp(current_day_economic):
            (
                gdp, unemployment, inflation, interest_rate,
                cpi, ppi, balance_of_trade, retail_sales, industrial_production,
                usd_price_euro,
                xlk, xlf, xlv, xly, xlp, xle, xli, xlb, xlre, xlu, xlc
            ) = self.fetch_economic_data(next_day_economic)
            
            # Create a new row of economic data
            new_entry_economic = [
                next_day_economic.strftime('%Y-%m-%d'), gdp, unemployment, inflation, interest_rate,
                cpi, ppi, balance_of_trade, retail_sales, industrial_production,
                usd_price_euro, xlk, xlf, xlv, xly, xlp, xle, xli, xlb, xlre, xlu, xlc
                # ... Add other indicators here
            ]
            
            # Append the new row to the DataFrame
            df_economic.loc[len(df_economic)] = new_entry_economic
            
            next_day_economic += timedelta(days=1)

            time.sleep(0.5)
        
        # Save the updated economic data DataFrame
        df_economic.to_csv('WorldEconomicData.csv', index=False)


nk = newsKeeper()
nk.updateNews()
nk.updateEconomicData()

