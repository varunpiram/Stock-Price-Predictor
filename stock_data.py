import yfinance as yf

class StockData:
    def __init__(self):
        return
    
    def getdata(self, ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        return data
    
    def generatecsv(self, ticker, start_date, end_date):
        data = self.getdata(ticker=ticker, start_date=start_date, end_date=end_date)
        data.to_csv(f"data/{ticker}.csv")

    def singledaydata(self, ticker, date):
        data = self.getdata(ticker=ticker, start_date=date, end_date=date)
        return data
    

sd = StockData()
sd.generatecsv(ticker="AAPL", start_date="2010-01-01", end_date="2020-01-01")







