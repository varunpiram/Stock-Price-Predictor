import pandas as pd
import yfinance as yf
import datetime
from fredapi import Fred


# Econ keeper class - updates the world economic data csv file
class econKeeper:

    # Initializes fred api and file path, gets last date in csv file
    def __init__(self, fred_api_key):
        self.file_path = 'WorldEconomicData.csv'
        self.fred = Fred(api_key=fred_api_key)
        self.last_date = self.get_last_date()

    # Gets last date in csv file
    def get_last_date(self):
        df = pd.read_csv(self.file_path)
        last_date_str = df['Date'].iloc[-1]
        return datetime.datetime.strptime(last_date_str, '%d-%b-%y').date()

    # Gets all days between two dates
    def get_all_days(self, start_date, end_date):
        return [start_date + datetime.timedelta(days=i) for i in range((end_date-start_date).days + 1)]

    # Fetches relevant data from FRED api
    def fetch_fred_data_range(self, start_date, end_date):
        # Series codes for data to fetch
        series_codes = ['T5YIE', 'FEDFUNDS', "SP500"]
        data = {}
        # Fetches data for each series code
        try:
            for series in series_codes:
                series_data = self.fred.get_series(series, start_date, end_date)
                data[series] = series_data
        except:
            print("Couldn't fetch data: fetch_fred_data_range")
            Exception("Couldn't fetch data")
        return data

    # Fetches relevant data from yfinance
    def fetch_yfinance_data_range(self, start_date, end_date):
        # Sector tickers (and USD in EUR)
        tickers = ["EUR=X", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC"]
        try:
            data = yf.download(tickers, start=start_date, end=end_date)
        except:
            print("Couldn't fetch data: fetch_yfinance_data_range")
            Exception("Couldn't fetch data")
        return data

    # Updates the csv file with new data
    def updateEcon(self):
        # Get today's date
        today = datetime.date.today()
        # Check to make sure there is something to update
        if self.last_date != today:

            # Gets the difference in days
            diff_days = (today - self.last_date).days
            
            # Caps the difference in days at 3000 (API limit safety)
            if diff_days > 3000:
                today = self.last_date + datetime.timedelta(days=3000)

            # Gets all days between last date and today (or 3000 days from last date)
            all_days = self.get_all_days(self.last_date, today)

            # Fetches data from FRED and yfinance for the date range
            yfinance_data = self.fetch_yfinance_data_range(self.last_date, today)
            fred_data_range = self.fetch_fred_data_range(self.last_date, today)
            
            # Fetches close data for the yfinance tickers
            yfinance_data_closes = yfinance_data.xs('Close', axis=1, level=0)

            new_rows = []

            # Loops through all days and adds the data to a new row
            for date in all_days:
                if date <= self.last_date:
                    continue
                
                ts_date = pd.Timestamp(date)
                # Gets FRED data for the date
                fred_data = {key: value[date] if date in value.index else None for key, value in fred_data_range.items()}
                # Gets yfinance data for the date
                yf_row = yfinance_data_closes.loc[ts_date].to_dict() if ts_date in yfinance_data_closes.index else {ticker: float('nan') for ticker in yfinance_data_closes.columns}
                # Creates new row with date atrribute
                row = {'Date': date.strftime('%d-%b-%y')}
                # Updates row with FRED and yfinance data
                row.update(fred_data)
                row.update(yf_row)
                # Adds row to new rows list
                new_rows.append(row)

            # Creates dataframe with the new rows
            df = pd.DataFrame(new_rows)

            # Turns current csv data into dataframe
            existing_df = pd.read_csv(self.file_path)

            # Adds new data's dataframe to current csv data dataframe
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Fills in NaN values
            combined_df.fillna(method='ffill', inplace=True)  # forward fill first to fill in any NaNs at the beginning
            combined_df.fillna(method='bfill', inplace=True)  # backward fill after forward fill to fill in any remaining NaNs

            # Gets columns in order
            columns_order = existing_df.columns.tolist()

            # Sorts columns in new dataframe
            combined_df = combined_df[columns_order]

            # Saves new dataframe to csv file
            combined_df.to_csv(self.file_path, mode='w', header=True, index=False)

        else:
            # If there is nothing to update, print message
            print("Economic data already up to date")





