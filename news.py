import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io
import requests
import re


# News keeper class - updates the world news data csv file
class newsKeeper:

    # Initializes class
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
    
    # Safely gets data
    def data_safe(self, data, ticker):
        try:
            return data[ticker].iloc[0]
        except:
            return None


    # Fetch GDELT's news information for a given date (GDELT mentions) by downloading the csv into
    # memory and extracting the URLS
    def fetch_gd(self, date):
        # Format date
        formatted_date = date.strftime('%Y%m%d')
        # Get URL for GDELT data
        gdelt_url = f"http://data.gdeltproject.org/gdeltv2/{formatted_date}000000.mentions.CSV.zip"
        
        # Download data
        response = requests.get(gdelt_url, stream=True)
        # Unzip data in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(f"{formatted_date}000000.mentions.CSV") as f:
                df = pd.read_csv(f, sep='\t', header=None)
        return df[5].tolist()
    
        
    # Updates WorldNewsData.csv with headlines up until the current day by fetching GDELT information,
    # extracting the titles from the URLs, and adding them to the csv file
    def updateNews(self):
        # Read world news data csv file
        df = pd.read_csv('WorldNewsData.csv')
        # Gets the latest entry
        last_date = pd.to_datetime(df['Date'].iloc[-1], format='%d-%b-%y')
        # Gets today's date
        current_day = datetime.now().date()

        # If the last entry is not today, update the csv file
        if last_date != pd.Timestamp(current_day):

            # Gets the next day to update
            next_day = last_date + timedelta(days=1)

            # Updates the csv file until it is up to date
            while next_day <= pd.Timestamp(current_day):

                # Gets the headline urls for the next day
                urls = self.fetch_gd(next_day)
                
                # Filter duplicate URLS
                unique_urls = list(set(urls))
                
                # Extract titles from URLs
                titles = [self.extract_title_from_url(url) for url in unique_urls][:25]
                new_entry = [next_day.strftime('%d-%b-%y')] + titles
                df.loc[len(df)] = new_entry
                next_day += timedelta(days=1)
            
            # Save the updated csv file
            df.to_csv('WorldNewsData.csv', index=False)
        else:
            print("News already up to date.")
 




