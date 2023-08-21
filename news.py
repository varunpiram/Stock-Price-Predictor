import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
import requests
import re



class newsKeeper:

    def __init__(self):
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
    
    def data_safe(self, data, ticker):
        try:
            return data[ticker].iloc[0]
        except:
            return None


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

 



nk = newsKeeper()
nk.updateNews()



