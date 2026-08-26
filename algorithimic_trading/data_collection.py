import yfinance as yf
import json
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timezone
def dataset(tickers):##collect data for the strategy
    data = yf.download(tickers,period="1y",interval="1d")
    data = data.to_json()
    with open("dataset.json","w") as file:
        json.dump(data,file,indent = 4)
def backtest_dataset(tickers):#out of sample testing data for the strategy
    data = yf.download(tickers,start="2020-01-01",end="2021-01-01",interval="1d")
    data = data.to_json()
    with open("backtest_data.json","w") as file:
        json.dump(data,file,indent = 4)
def backtest_comparison():#collect data to compare the strategy to the S&P 500 index
    data = yf.download("^GSPC",start = "2020-01-01",end = "2021-01-01",interval = "1d")
    data = data[["Open","Close"]]
    data.columns = ["Open","Close"]
    data = json.loads(data.to_json())
    with open("backtest_comparison.json","w") as file:
        json.dump(data,file,indent = 4)
    daily_change = percentage_change("backtest_comparison.json")
    return daily_change


def ticker_request():#scrapes wiki for top 100 tickers
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))

    sp500 = tables[0]["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in sp500[:100]]
    return tickers
def restructure_dataset(dataset_path):
    with open(dataset_path) as f:
        raw = f.read()

    data = json.loads(json.loads(raw))

    def parse_key(key):#removes unnecessary punctuation to increase readability
        key = key.strip("()").replace("'", "")
        parts = key.split(", ")
        return parts[0], parts[1]

    def ms_to_date(ms):#converts to date
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    structured = {}

    for raw_key, date_values in data.items():
        field, ticker = parse_key(raw_key)

        if ticker not in structured:#creates ticker index
            structured[ticker] = {}

        for timestamp, value in date_values.items():
            date = ms_to_date(timestamp)

            if date not in structured[ticker]:
                structured[ticker][date] = {}#creates new date index

            structured[ticker][date][field] = value# adds values to date

    for ticker in structured:
        structured[ticker] = dict(sorted(structured[ticker].items()))

    # Save to new file
    with open(dataset_path, "w") as f:
        json.dump(structured, f, indent=4)
def moving_average(dataset):#creates 5 day rolling average feature
    if dataset == "dataset.json":
        with open("dataset.json","r") as file:
            data = json.load(file)
            i = 0 
            ma = []
            for ticker in data:
                for date in data[ticker]:
                    dates= list(data[ticker].keys())
                    for i in range (len(dates)):            
                        if i >= 5:
                            ma = (data[ticker][date]["Close"]+data[ticker][dates[i-1]]["Close"]+data[ticker][dates[i-2]]["Close"]+data[ticker][dates[i-3]]["Close"]+data[ticker][dates[i-4]]["Close"])/5
                        
                        else:
                            ma = 0
                    data[ticker][date]["moving_average"] = ma
        with open ("dataset.json","w") as file:
            json.dump(data, file,indent = 4 )
    elif dataset == "backtest_data.json":
        with open("backtest_data.json","r") as file:
            data = json.load(file)
            i = 0 
            ma = []
            for ticker in data:
                for date in data[ticker]:
                    dates= list(data[ticker].keys())
                    for i in range (len(dates)):            
                        if i >= 5:
                            ma = (data[ticker][date]["Close"]+data[ticker][dates[i-1]]["Close"]+data[ticker][dates[i-2]]["Close"]+data[ticker][dates[i-3]]["Close"]+data[ticker][dates[i-4]]["Close"])/5
                        
                        else:
                            ma = 0
                    data[ticker][date]["moving_average"] = ma
        with open ("backtest_data.json","w") as file:
            json.dump(data, file,indent = 4 )
def percentage_change(dataset):#creates percentage change feature
    with open(dataset,"r") as file:
        data = json.load(file)
        close_p = 0.0
        open_p = 0.0
        if dataset == "dataset.json":
            for ticker in data:
                for date in data[ticker]:
                    close_p = data[ticker][date]["Close"]    
                    open_p = data[ticker][date]["Open"]
                    percentage_change = ((close_p - open_p)/open_p)
                    data[ticker][date]["percentage_change"] = percentage_change
            with open (dataset,"w") as file:
                json.dump(data, file,indent = 4 )
        if dataset == "backtest_data.json":
            for ticker in data:
                for date in data[ticker]:
                    close_p = data[ticker][date]["Close"]    
                    open_p = data[ticker][date]["Open"]
                    percentage_change = ((close_p - open_p)/open_p)
                    data[ticker][date]["percentage_change"] = percentage_change
            with open (dataset,"w") as file:
                json.dump(data, file,indent = 4 )
        elif dataset == "backtest_comparison.json":
            open_p = list(data["Open"].values())
            close_p = list(data["Close"].values())
            percentage_change = []
            for i in range (len(open_p)):
                percentage_change.append(((close_p[i] - open_p[i])/open_p[i]))
            return percentage_change
