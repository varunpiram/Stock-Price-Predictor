from model import StockPredictor
from news import newsKeeper
from econ import econKeeper
from data import dataGenerator
from dotenv import load_dotenv
import os

load_dotenv()

# Gets API key
FRED_KEY = os.environ.get("FRED_KEY")

class app:

    def __init__(self):
        pass

    # Updates data and trains the model based on user-inputted hyperparameters
    def train(self, ticker):
        epochs = int(input("Enter epochs (rec. 50): "))
        batch_size = int(input("Enter batch size (rec. 32): "))

        print("Updating news and economic data...")
        news = newsKeeper()
        news.updateNews()

        econ = econKeeper(FRED_KEY)
        econ.updateEcon()
        print("Finished updating news and economic data.")

        print("Generating data...")
        dg = dataGenerator()
        dg.score(ticker)
        print("Finished generating data.")

        print("Training model...")
        mdl = StockPredictor(ticker)
        mdl.train_model(epochs=epochs, batch_size=batch_size)
        print("Finished training model.")

    # Allows the user to run models and retrain - lets users run model on specific dates, display test
    # data performance, run backtests over custom date range, and retrain the model w/o updating data
    def use(self, ticker):

        mdl = StockPredictor(ticker)
        
        while True:
            print("Select an option (1/2/3/4/5):")
            print("1. Predict Specific Date")
            print("2. Display Test Performance")
            print("3. Backtest")
            print("4. Retrain (No data updates)")
            print("5. Exit")

            choice = input("Enter option: ")

            if choice == '1':
                try:
                    date = input("Enter date (YYYY-MM-DD): ")
                    print(mdl.predict_given_date(date))
                except:
                    print("Invalid date. Try again.")
            elif choice == '2':
                mdl.predict_test_data()
            elif choice == '3':
                date = input("Enter start date (YYYY-MM-DD): ")
                date1 = input("Enter end date (YYYY-MM-DD): ")
                mdl.backtest(date, date1)
            elif choice == '4':
                epochs = int(input("Enter epochs (rec. 50): "))
                batch_size = int(input("Enter batch size (rec. 32): "))
                print("Training model...")
                mdl.train_model(epochs=epochs, batch_size=batch_size)
                print("Finished training model.")
            elif choice == '5':
                break
            else:
                print("Invalid option. Try again.")


# Main method - allows user to enter a ticker and train or use the model
    def main(self):
         while True:
            ticker = input("Enter ticker ('exit' to quit): ")
            if ticker.lower() == "exit":
                break
            else:
                ticker = ticker.upper()
                print("Select an option (1/2/3):")
                print("1. Train (Updates New Data)")
                print("2. Run (Uses Existing Data)")
                print("3. Exit")
                choice = input("Enter option: ")

                if choice == "1":
                    self.train(ticker)
                elif choice == "2":
                    if not os.path.exists(f"data/Model_{ticker}.h5"):
                        print("Model does not exist. Train first.")
                        break
                    if not os.path.exists(f"data/Data_{ticker}.csv"):
                        print("Data does not exist. Train first.")
                        break
                    self.use(ticker)
                elif choice == "3":
                    break
                else:
                    print("Invalid option. Try again.")
                

ins = app()
ins.main()

            

