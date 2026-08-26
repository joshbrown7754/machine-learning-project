

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import json
from io import StringIO
from datetime import datetime, timezone
import pyqtgraph as pg
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr



def dataset(tickers):
    data = yf.download(tickers,period="1y",interval="1d")
    data = data.to_json()
    with open("dataset.json","w") as file:
        json.dump(data,file,indent = 4)
def ticker_request():
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))

    sp500 = tables[0]["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in sp500[:100]]
    return tickers
def backtest_comparison():
    data = yf.download("^GSPC",start = "2020-01-01",end = "2021-01-01",interval = "1d")
    data = data[["Open","Close"]]
    data.columns = ["Open","Close"]
    data = json.loads(data.to_json())
    with open("backtest_comparison.json","w") as file:
        json.dump(data,file,indent = 4)
    daily_change = percentage_change("backtest_comparison.json")
    return daily_change
def backtest_dataset(tickers):
    data = yf.download(tickers,start="2020-01-01",end="2021-01-01",interval="1d")
    data = data.to_json()
    with open("backtest_data.json","w") as file:
        json.dump(data,file,indent = 4)
def restructure_dataset(dataset_path):
    # Load the dataset (it's stored as a JSON string inside JSON)
    with open(dataset_path) as f:
        raw = f.read()

    data = json.loads(json.loads(raw))

    # Parse the flat keys like "('Close', 'AAPL')" into (fiecold, ticker)
    def parse_key(key):
        # key looks like "('Close', 'AAPL')"
        key = key.strip("()").replace("'", "")
        parts = key.split(", ")
        return parts[0], parts[1]

    # Convert millisecond timestamp to readable date string
    def ms_to_date(ms):
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    # Restructure into: { "AAPL": { "2026-05-01": { "Open": x, "High": x, ... }, ... }, ... }
    structured = {}

    for raw_key, date_values in data.items():
        field, ticker = parse_key(raw_key)

        if ticker not in structured:
            structured[ticker] = {}

        for timestamp, value in date_values.items():
            date = ms_to_date(timestamp)

            if date not in structured[ticker]:
                structured[ticker][date] = {}

            structured[ticker][date][field] = value

    # Sort each ticker's dates chronologically
    for ticker in structured:
        structured[ticker] = dict(sorted(structured[ticker].items()))

    # Save to new file
    with open(dataset_path, "w") as f:
        json.dump(structured, f, indent=4)

def moving_average(dataset):
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

def max(datapoint):
    with open("dataset.json","r") as file:
        data = json.load(file)
        high = 0
        for ticker in data:
            for dates in data[ticker]:
                if datapoint == "Volume":
                    if np.log10(data[ticker][dates][datapoint]) >= high:
                        high = np.log10(data[ticker][dates][datapoint])
                else:
                    if data[ticker][dates][datapoint] >= high:
                        high = data[ticker][dates][datapoint]
                
        return high
def min(datapoint):
    with open("dataset.json","r") as file:
        data = json.load(file)
        low = 999999999999
        for ticker in data:
            for dates in data[ticker]:
                if datapoint == "Volume":
                    if np.log10(data[ticker][dates][datapoint]) <= low:
                        low = np.log10(data[ticker][dates][datapoint])
                else:
                    if data[ticker][dates][datapoint] <= low:
                        low = data[ticker][dates][datapoint]
        return low

def normalize(x,max,min,):
    normalized = (x-min)/(max-min)
    return normalized
def node_calc(inputs,weight,bias,final_out):
    if final_out == True:
        outputs = []
        for i in range (len(inputs)):
            temp = inputs[i]*weight[i]
            outputs.append(temp)
        output = sum(outputs)
        return(output)
    elif isinstance(inputs,dict):
        x1 = inputs["op"]*weight["op"]
        x2 = inputs["cl"]*weight["cl"]
        x3 = inputs["vo"]*weight["vo"]
        x4 = inputs["hi"]*weight["hi"]
        x5 = inputs["lo"]*weight["lo"]
        x6 = inputs["pc"]*weight["pc"]
        x7 = inputs["ma"]*weight["ma"]
        output = x1+x2+x3+x4+x5+x6+x7+bias
        if output <= 0 :
            return 0
        else:
            return output
    elif isinstance(inputs,list):
        outputs = []
        for i in range (len(inputs)):
            temp = inputs[i]*weight[i]
            outputs.append(temp)
        output = sum(outputs)
        if output <= 0 :
            return 0
        else:
            return output
def percentage_change(dataset):
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
    
    
def layer1(inputs):
    bias = 0
    node1  ={
        "op": 0.21, 
        "cl": -0.65,
        "vo": 0.48,
        "hi": -0.12,
        "lo": 0.73,
        "pc": -0.34,
        "ma": 0.09
    }
    node2  ={
        "op": 0.51, 
        "cl": -0.22,
        "vo": 0.13,
        "hi": -0.68,
        "lo": 0.72,
        "pc": -0.05,
        "ma": 0.34
        }
    node3  ={
            "op": -0.61, 
            "cl": 0.44,
            "vo": -0.09,
            "hi": 0.27,
            "lo": -0.73,
            "pc": 0.66,
            "ma": -0.18
            }
    node4  ={
            "op": 0.08, 
            "cl": -0.57,
            "vo": 0.69,
            "hi": -0.31,
            "lo": 0.25,
            "pc": -0.71,
            "ma": 0.47
            }
    x1 = node_calc(inputs,node1,bias,False) 
    x2 = node_calc(inputs,node2,bias,False)
    x3 = node_calc(inputs,node3,bias,False)
    x4 = node_calc(inputs,node4,bias,False)
    outputs = [x1,x2,x3,x4]
    return layer2(outputs)
def layer2(inputs):
    bias = 0
    node1 = [0.842, -0.317, 1.021, -0.664]
    node2 = [-0.903, 0.558, -0.112, 0.974]
    node3 = [0.129, -1.044, 0.687, -0.256]
    node4 = [-0.771, 0.334, -0.598, 1.063]
    x1 = node_calc(inputs,node1,bias,False) 
    x2 = node_calc(inputs,node2,bias,False)
    x3 = node_calc(inputs,node3,bias,False)
    x4 = node_calc(inputs,node4,bias,False)
    outputs = [x1,x2,x3,x4]
    return layer3(outputs)
def layer3(inputs):
    bias = 0
    node1 = [0.582, -0.913, 0.274, -0.661]
    node2 = [-0.348, 0.799, -0.125, 0.936]
    x1 = node_calc(inputs,node1,bias,False) 
    x2 = node_calc(inputs,node2,bias,False)
    outputs = [x1,x2,]
    return layer4(outputs)
def layer4(inputs):
    bias = 0
    node1 = [1.128, -0.734]
    x1 = node_calc(inputs,node1,bias,True) 
    outputs = x1
    return outputs
def prediction_neural_net(training = True,dataset = "dataset.json"):
    with open("dataset.json","r") as file:
            data = json.load(file)
            learning_rate = 0.01
            maximums = []
            minimums = []

            datapoints = ["Open","Close","Volume","High","Low","moving_average"]
            for i in range (len(datapoints)):
                maximums.append(max(datapoints[i]))
                minimums.append(min(datapoints[i]))
    if training == True:
            for ticker in data:
                for date in data[ticker]:
                    dates= list(data[ticker].keys())
                    if i != len(dates) - 1:
                        targetdate = dates[i+1]
                        target = data[ticker][targetdate]["percentage_change"]
                
                    inputs = {
                        "op" : normalize(float(data[ticker][date]["Open"]),maximums[0],minimums[0]),
                        "cl" : normalize(float(data[ticker][date]["Close"]),maximums[1],minimums[1]),
                        "vo" : normalize(np.log10(float(data[ticker][date]["Volume"])),maximums[2],minimums[2]),
                        "hi" : normalize(float(data[ticker][date]["High"]),maximums[3],minimums[3]),
                        "lo" : normalize(float(data[ticker][date]["Low"]),maximums[4],minimums[4]),
                        "pc" : float(data[ticker][date]["percentage_change"]),
                        "ma" : normalize((data[ticker][date]["moving_average"]),maximums[5],minimums[5])
                    }
                    final_output = layer1(inputs)
                    loss = (final_output-target)**2
                    #print("final_output:",final_output)
                    #print("loss:",loss)
    if training == False:
        outputs = []
        choice = []
        with open(dataset,"r") as file:
            data = json.load(file)
            for ticker in data:
                for date in data[ticker]:
                    inputs = {
                        "op" : normalize(float(data[date][ticker]["Open"]),maximums[0],minimums[0]),
                        "cl" : normalize(float(data[date][ticker]["Close"]),maximums[1],minimums[1]),
                        "vo" : normalize(np.log10(float(data[date][ticker]["Volume"])),maximums[2],minimums[2]),
                        "hi" : normalize(float(data[date][ticker]["High"]),maximums[3],minimums[3]),
                        "lo" : normalize(float(data[date][ticker]["Low"]),maximums[4],minimums[4]),
                        "pc" : float(data[date][ticker]["percentage_change"]),
                        "ma" : normalize((data[date][ticker]["moving_average"]),maximums[5],minimums[5])
                    }
                    final_output = layer1(inputs)
                    outputs.append(final_output)
                    outputs = bubble_sort(outputs)
                    choice.append(outputs[0])
                    
            return choice       
                   
tickers = ticker_request()
mode = int(input("what mode do you want to run the program in? (1) training or (2) out of sample testing or (3) prediction: "))
if mode == 1:
    dataset(tickers)
    dataset_path = r"D:\algorithimic trading\dataset.json"
    restructure_dataset(dataset_path)
    moving_average("dataset.json")
    percentage_change("dataset.json")
    prediction_neural_net()
elif mode == 2:
    sp500_balance = 10000
    model_balance = 10000
    backtest_dataset(tickers)
    dataset_path = r"D:\algorithimic trading\backtest_data.json"
    restructure_dataset(dataset_path)
    backtest_comparison()
    moving_average( "backtest_data.json")
    percentage_change("backtest_data.json")
    app = pg.mkQApp("simple window")

    win = pg.plot(title="sp 500 vs neauralnet prediction")
    win.setWindowTitle("sp 500 vs neauralnet prediction")
    win.setLabel('left', 'Balance', units='$')
    win.setLabel('bottom', 'Days')
    win.plot(sp500_balance,pen = "g")
    win.plot(model_balance,pen = "b")
    
    neural_net_prediction = prediction_neural_net(training = False,dataset = "backtest_data.json")
    sp500_daily_change = backtest_comparison()
    for i in range (len(neural_net_prediction)):
        model_balance = model_balance * (1+neural_net_prediction[i])
        sp500_balance = sp500_balance * (1+sp500_daily_change[i])
        
        win.plot(sp500_balance,pen = "g")
        win.plot(model_balance,pen = "b")
    