
import datetime
import csv

class SentimentData:

    def __init__(self):
        return
    
    # Converts a given date in format YYYY-MM-DD to an index in the csv file and returns headlines for
    # that date as a string. Doesn't search through csv, just finds the line.
    # DO NOT MODIFY WorldNewsData.csv OR THIS METHOD WILL NOT WORK.
    def gethistoricheadlines(self, date):
        
        ind = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime(2018, 5, 1)).days + 2

        if 1 <= ind <= 1860:
            with open("WorldNewsData.csv", "r", encoding="utf-8") as file:
                reader = csv.reader(file)

                for _ in range(ind):
                    row = next(reader)

                headlines = [col.strip() for col in row[1:] if col.strip()]
                return ' '.join(headlines)
            
        else:
            raise Exception("Date out of range")
        
    
sd = SentimentData()
print(sd.gethistoricheadlines("2018-09-05"))








    