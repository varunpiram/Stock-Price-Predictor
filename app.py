from model import StockPredictor
from news import newsKeeper
from econ import econKeeper
from data import dataGenerator
from dotenv import load_dotenv
import os

load_dotenv()

FRED_KEY = os.environ.get("FRED_KEY")

# Easy way to run model via. command line
class app:
    # Initializes app
    def __init__(self):
        return
    
    # Runs app
    def run(self, scoreToggle=True):

        # Updates common data shared across all tickers (WorldEconomicData and WorldNewsData)
        print("Updating common data...")
        try:
            news = newsKeeper()
            news.updateNews()
            econ = econKeeper(FRED_KEY)
            econ.updateEcon()
            print("Common data updated.")
        except:
            print("Couldn't update common data. Check World CSV files and internet connection.")

        # Asks ticker for model
        ticker = input("Enter your ticker: ")

        # Lets user update data
        while True:
            ask0 = input(f"Update Data for {ticker}? (Yes/No): ")
            if ask0 == "Yes":
                try:
                    print("Updating data...")
                    data = dataGenerator()
                    data.score(ticker, sentimentScore=scoreToggle)
                    print(f"Data for {ticker} updated.")
                except:
                    print("Couldn't update data. Check ticker validity and internet connection. ")
                break
            if ask0 == "No":
                if os.path.exists(f"data/Data_{ticker}.csv"):
                    break
                else:
                    ask00 = input(f"No data for {ticker} exists, do you wish to proceed? (Yes/No): ")
                    if ask00 == "Yes":
                        break
                    if ask00 == "No":
                        continue
                    else:
                        print("Invalid input. Answer must be 'Yes' or 'No'.")
            else:
                print("Invalid input. Answer must be 'Yes' or 'No'.")

        # Lets user train model
        while True:
            ask1 = input("Train model? (Yes/No): ")
            if ask1 == "Yes":
                if os.path.exists(f"data/Model_{ticker}.h5"):
                    print("Training model...")
                    model = StockPredictor(ticker)
                    model.train_model(lookback=1, epochs=50, batch_size=32)
                    print("Model trained.")
                else:
                    print("Model doesn't exist. Creating model...")
                    lookback1 = int(input("Enter lookback (suggested to use 1): "))
                    epochs1 = int(input("Enter epochs (suggested to use 50): "))
                    batch_size1 = int(input("Enter batch size (suggested to use 32): "))
                    model = StockPredictor(ticker)
                    model.train_model(lookback=lookback1, epochs=epochs1, batch_size=batch_size1)
                    print("Model created.")
                break
            if ask1 == "No":
                if os.path.exists(f"data/Model_{ticker}.h5"):
                    break
                else:
                    ask10 = input(f"No model for {ticker} exists, do you wish to proceed? (Yes/No): ")
                    if ask10 == "Yes":
                        break
                    if ask10 == "No":
                        continue
                    else:   
                        print("Invalid input. Answer must be 'Yes' or 'No'.")

            else:
                print("Invalid input. Answer must be 'Yes' or 'No'.")
        
        # Lets user use model - predicts next day high for specific dates, displays test data
        # performance as a graph, and performs backtests over custom ranges
        while True:
            print("Select an option:")
            print("1. Predict next day high")
            print("2. Display test performance")
            print("3. Perform backtest")
            print("4. Exit")
            ask2 = input("Enter your choice (1/2/3/4): ")

            if ask2 == "1":
                if os.path.exists(f"data/Model_{ticker}.h5"):
                    ask20 = input("Enter day to predict next day high (YYYY-MM-DD): ")
                    try:
                        print(f"Predicted high for {ask20} is {model.predict_given_date(ask20)}")
                    except:
                        print("Invalid date.")
                else:
                    print(f"No model for {ticker} exists. Please train model first.")
                    continue
                    
            if ask2 == "2":
                if os.path.exists(f"data/Model_{ticker}.h5"):
                    model.predict_test_data()
                else:
                    print(f"No model for {ticker} exists. Please train model first.")
                    continue
            
            if ask2 == "3":
                if os.path.exists(f"data/Model_{ticker}.h5"):
                    ask20 = input("Enter backtest starting date (YYYY-MM-DD): ")
                    ask21 = input("Enter backtest ending date (YYYY-MM-DD): ")
                    try:
                        model.backtest(start_date=ask20, end_date=ask21)
                    except:
                        print("Invalid date range.")
                else:
                    print(f"No model for {ticker} exists. Please train model first.")
                    continue
            
            if ask2 == "4":
                print("Exiting...")
                break
            else:
                print("Invalid input. Response must be '1', '2', '3' or '4'.")

            

# Runs app
inst = app()
inst.run()
