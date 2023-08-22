import pandas as pd
import yfinance as yf
import datetime
from fredapi import Fred


#TODO: ADD ENVIRONMENTAL VARIABLE STUFF FOR API KEY!!!! API KEY HARDCODED RIGHT NOW!!

class EconomicDataUpdater:
    def __init__(self, fred_api_key):
        self.file_path = 'WorldEconomicData.csv'
        self.fred = Fred(api_key=fred_api_key)
        self.last_date = self.get_last_date()

    def get_last_date(self):
        df = pd.read_csv(self.file_path)
        last_date_str = df['Date'].iloc[-1]
        return datetime.datetime.strptime(last_date_str, '%d-%b-%y').date()

    def get_all_days(self, start_date, end_date):
        return [start_date + datetime.timedelta(days=i) for i in range((end_date-start_date).days + 1)]

    def fetch_fred_data_range(self, start_date, end_date):
        series_codes = ['T5YIE', 'FEDFUNDS', "SP500"]
        data = {}
        for series in series_codes:
            series_data = self.fred.get_series(series, start_date, end_date)
            series_data.index = series_data.index.date
            data[series] = series_data
        return data

    def fetch_yfinance_data_range(self, start_date, end_date):
        tickers = ["EUR=X", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC"]
        data = yf.download(tickers, start=start_date, end=end_date)
        return data

    def update_data(self):
        today = datetime.date.today()
        diff_days = (today - self.last_date).days
        if diff_days > 1000:
            today = self.last_date + datetime.timedelta(days=1000)

        all_days = self.get_all_days(self.last_date, today)

        yfinance_data = self.fetch_yfinance_data_range(self.last_date, today)
        fred_data_range = self.fetch_fred_data_range(self.last_date, today)
        
        # Extract only the 'Close' values for each ticker
        yfinance_data_closes = yfinance_data.xs('Close', axis=1, level=0)

        new_rows = []
        for date in all_days:
            if date <= self.last_date:
                continue

            ts_date = pd.Timestamp(date)

            # Get FRED data for the date. If not available, set as None
            fred_data = {key: value[date] if date in value.index else None for key, value in fred_data_range.items()}

            # Get yfinance data for the date. If not available, set as NaN (this is Pandas' default for missing data)
            yf_row = yfinance_data_closes.loc[ts_date].to_dict() if ts_date in yfinance_data_closes.index else {ticker: float('nan') for ticker in yfinance_data_closes.columns}

            row = {'Date': date.strftime('%d-%b-%y')}
            row.update(fred_data)
            row.update(yf_row)
            new_rows.append(row)

        df = pd.DataFrame(new_rows)

        # Load the existing DataFrame
        existing_df = pd.read_csv(self.file_path)

        # Append the new rows to the existing DataFrame
        combined_df = pd.concat([existing_df, df], ignore_index=True)

        # Forward fill and back fill missing data
        combined_df.fillna(method='ffill', inplace=True)  # forward fill
        combined_df.fillna(method='bfill', inplace=True)  # backward fill after forward fill to ensure all NaNs are filled

        columns_order = existing_df.columns.tolist()  # Get the existing column order

        # Reorder the DataFrame columns to match the existing order
        combined_df = combined_df[columns_order]

        # Save the updated DataFrame back to the CSV
        combined_df.to_csv(self.file_path, mode='w', header=True, index=False)







# Usage
updater = EconomicDataUpdater('1353cc07f1bd42904e62f96d8f984d86')
updater.update_data()
