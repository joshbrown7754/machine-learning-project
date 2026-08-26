import json
import torch 
import numpy as np
import pyqtgraph as pg
from collections import deque
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

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
        output = sum(outputs)+bias
        return(output)
    else:
        outputs = []
        for i in range (len(inputs)):
            temp = inputs[i]*weight[i]+bias
            outputs.append(temp)
        output = sum(outputs)
        output = torch.nn.functional.leaky_relu(output, negative_slope=0.01)
        return output

def input_layer(inputs,weights,nodes):
    bias = torch.tensor([
        weights["input_layer"]["bias"]["b0"],
        weights["input_layer"]["bias"]["b1"],
        weights["input_layer"]["bias"]["b2"],
        weights["input_layer"]["bias"]["b3"],
        ],dtype=torch.float32,requires_grad=True)
    node1  = torch.tensor([
        weights["input_layer"]["node1"]["w0"],
        weights["input_layer"]["node1"]["w1"],
        weights["input_layer"]["node1"]["w2"],
        weights["input_layer"]["node1"]["w3"],
        weights["input_layer"]["node1"]["w4"],
        weights["input_layer"]["node1"]["w5"],
        weights["input_layer"]["node1"]["w6"],
    ],requires_grad=True)
    node2  = torch.tensor([
        weights["input_layer"]["node2"]["w0"],
        weights["input_layer"]["node2"]["w1"],
        weights["input_layer"]["node2"]["w2"],
        weights["input_layer"]["node2"]["w3"],
        weights["input_layer"]["node2"]["w4"],
        weights["input_layer"]["node2"]["w5"],
        weights["input_layer"]["node2"]["w6"]
        ],requires_grad=True)
    node3  = torch.tensor([
        weights["input_layer"]["node3"]["w0"],
        weights["input_layer"]["node3"]["w1"],
        weights["input_layer"]["node3"]["w2"],
        weights["input_layer"]["node3"]["w3"],
        weights["input_layer"]["node3"]["w4"],
        weights["input_layer"]["node3"]["w5"],
        weights["input_layer"]["node3"]["w6"]    
        ],requires_grad=True)
    node4  = torch.tensor([
        weights["input_layer"]["node4"]["w0"],
        weights["input_layer"]["node4"]["w1"],
        weights["input_layer"]["node4"]["w2"],
        weights["input_layer"]["node4"]["w3"],
        weights["input_layer"]["node4"]["w4"],
        weights["input_layer"]["node4"]["w5"],
        weights["input_layer"]["node4"]["w6"]
        ],requires_grad=True)
    x1 = node_calc(inputs,node1,bias[0],False) 
    x2 = node_calc(inputs,node2,bias[1],False)
    x3 = node_calc(inputs,node3,bias[2],False)
    x4 = node_calc(inputs,node4,bias[3],False)
    outputs = [x1,x2,x3,x4]
    nodes["input_layer"] = {"bias":bias,
                            "node1":node1,
                            "node2":node2,
                            "node3":node3,
                            "node4":node4
                            }
    return layer2(outputs,weights,nodes)
def layer2(inputs,weights,nodes):
    bias = torch.tensor([
        weights["layer2"]["bias"]["b0"],
        weights["layer2"]["bias"]["b1"],
        weights["layer2"]["bias"]["b2"],
        weights["layer2"]["bias"]["b3"],
        ],dtype=torch.float32,requires_grad=True)
    node1 = torch.tensor([
        weights["layer2"]["node1"]["w0"],
        weights["layer2"]["node1"]["w1"],
        weights["layer2"]["node1"]["w2"],
        weights["layer2"]["node1"]["w3"]
    ],requires_grad=True)
    node2 = torch.tensor([
        weights["layer2"]["node2"]["w0"],
        weights["layer2"]["node2"]["w1"],
        weights["layer2"]["node2"]["w2"],
        weights["layer2"]["node2"]["w3"]
    ],requires_grad=True)
    node3 = torch.tensor([
        weights["layer2"]["node3"]["w0"],
        weights["layer2"]["node3"]["w1"],
        weights["layer2"]["node3"]["w2"],
        weights["layer2"]["node3"]["w3"]
    ],requires_grad=True)
    node4 = torch.tensor([
        weights["layer2"]["node4"]["w0"],
        weights["layer2"]["node4"]["w1"],
        weights["layer2"]["node4"]["w2"],
        weights["layer2"]["node4"]["w3"]
    ],requires_grad=True)
    x1 = node_calc(inputs,node1,bias[0],False) 
    x2 = node_calc(inputs,node2,bias[1],False)
    x3 = node_calc(inputs,node3,bias[2],False)
    x4 = node_calc(inputs,node4,bias[3],False)
    outputs = [x1,x2,x3,x4]
    nodes["layer2"] = {"bias":bias,
                        "node1":node1,
                        "node2":node2,
                        "node3":node3,
                        "node4":node4
                        }
    return layer3(outputs,weights,nodes)
def layer3(inputs,weights,nodes):
    bias = torch.tensor([
         weights["layer3"]["bias"]["b0"],
         weights["layer3"]["bias"]["b1"],
         weights["layer3"]["bias"]["b2"],
         weights["layer3"]["bias"]["b3"],
        ],dtype=torch.float32,requires_grad=True)    
    node1 = torch.tensor([
        weights["layer3"]["node1"]["w0"],
        weights["layer3"]["node1"]["w1"],
        weights["layer3"]["node1"]["w2"],
        weights["layer3"]["node1"]["w3"]
    ],requires_grad=True)
    node2 = torch.tensor([
        weights["layer3"]["node2"]["w0"],
        weights["layer3"]["node2"]["w1"],
        weights["layer3"]["node2"]["w2"],
        weights["layer3"]["node2"]["w3"]
    ],requires_grad=True)
    node3 = torch.tensor([
        weights["layer3"]["node3"]["w0"],
        weights["layer3"]["node3"]["w1"],
        weights["layer3"]["node3"]["w2"],
        weights["layer3"]["node3"]["w3"]
    ],requires_grad=True)
    node4 = torch.tensor([
        weights["layer3"]["node4"]["w0"],
        weights["layer3"]["node4"]["w1"],
        weights["layer3"]["node4"]["w2"],
        weights["layer3"]["node4"]["w3"]
    ],requires_grad=True)
    x1 = node_calc(inputs,node1,bias[0],False) 
    x2 = node_calc(inputs,node2,bias[1],False)
    x3 = node_calc(inputs,node3,bias[2],False)
    x4 = node_calc(inputs,node4,bias[3],False)
    outputs = [x1,x2,x3,x4]
    nodes["layer3"] = {"bias":bias,
                        "node1":node1,
                        "node2":node2,
                        "node3":node3,
                        "node4":node4
                        }
    return output_layer(outputs,weights,nodes)
def output_layer(inputs,weights,nodes):
    bias = torch.tensor(weights["output_layer"]["bias"]["b0"],dtype=torch.float32,requires_grad=True)
    node1 = torch.tensor([
        weights["output_layer"]["node1"]["w0"],
        weights["output_layer"]["node1"]["w1"],
        weights["output_layer"]["node1"]["w2"],
        weights["output_layer"]["node1"]["w3"]
    ],requires_grad=True)
    output = node_calc(inputs,node1,bias,True)    
    nodes["output_layer"] = {"node1":node1,"bias":bias}
    return output,nodes
def backpropagation(nodes,weights):
    learning_rate = 0.1
    #input layer weight updates
    
    for i in range (len(weights["input_layer"]["bias"])):
        j =str(i)
        bias = weights["input_layer"]["bias"]["b"+j] - learning_rate*(nodes["input_layer"]["bias"].grad[i])
        weights["input_layer"]["bias"]["b"+j] = float(bias)
    for nds in weights["input_layer"]:#nds refers to nodes in dictionary
        if nds == "bias":
            continue
        else:
            
            for i in range(len(weights["input_layer"][nds])):
                j=str(i)
                weight = weights["input_layer"][nds]["w"+j] - (learning_rate*nodes["input_layer"][nds].grad[i])
                weights["input_layer"][nds]["w"+j] = float(weight)
    #layer 2 weight updates
    
    for i in range (len(weights["layer2"]["bias"])):
        j =str(i)
        bias = weights["layer2"]["bias"]["b"+j] - (learning_rate*nodes["layer2"]["bias"].grad[i])
        weights["layer2"]["bias"]["b"+j] = float(bias)
    for nds in weights["layer2"]:
        if nds == "bias":
            continue
        else:
            for i in range(len(weights["layer2"][nds])):
                j=str(i)
                weight = weights["layer2"][nds]["w"+j] - (learning_rate*nodes["layer2"][nds].grad[i])
                weights["layer2"][nds]["w"+j] = float(weight)
    #layer 3 weight updates
    
    for i in range(len(weights["layer3"]["bias"])):
        j =str(i)
        bias = weights["layer3"]["bias"]["b"+j] - (learning_rate*nodes["layer3"]["bias"].grad[i])
        weights["layer3"]["bias"]["b"+j] = float(bias)
    for nds in weights["layer3"]:
        if nds == "bias":
            continue
        else:
            for i in range(len(weights["layer3"][nds])):
                j= str(i)
                weight = weights["layer3"][nds]["w"+j] - (learning_rate*nodes["layer3"][nds].grad[i])
                weights["layer3"][nds]["w"+j] = float(weight)
    #output layer
    bias  = weights["output_layer"]["bias"]["b0"] - learning_rate*(nodes["output_layer"]["bias"].grad)
    weights["output_layer"]["bias"]["b0"] = float(bias)
    for nds in weights["output_layer"]:
        if nds == "bias":
            continue
        else:
            for i in range(len(weights["output_layer"][nds])):
                j = str(i)
                weight = weights["output_layer"][nds]["w"+j] - learning_rate*(nodes["output_layer"][nds].grad[i])
                weights["output_layer"][nds]["w"+j] = float(weight)
    
def prediction_neural_net(training = True,dataset = "dataset.json"):
    with open("dataset.json","r") as file:
            data = json.load(file)
            maximums = []
            minimums = []

            datapoints = ["Open","Close","Volume","High","Low","moving_average"]
            for i in range (len(datapoints)):
                maximums.append(max(datapoints[i]))
                minimums.append(min(datapoints[i]))
    with open ("weights.json","r") as file:
        weights = json.load(file)
    if training == True:
        nodes  = {
            "input_layer"  : "",
            "layer2"       : "",
            "layer3"       : "",
            "output_layer" : ""  
        }
        avg_losses = deque(weights["graphdata"]["avg_losses"],maxlen=2000)
        steps = deque(weights["graphdata"]["steps"],maxlen=2000)
        avg_baseline_error = deque(weights["graphdata"]["avg_baseline_error"],maxlen=2000)
        epoch = weights["epoch"]
        temp = 0
        losses = []
        baseline_error = []
        step = 0
        app =pg.mkQApp()
        win = pg.plot(title="training losses graph")
        win.setWindowTitle("avg losses over amount of steps")
        label = pg.QtWidgets.QLabel("Loss: 0", win)
        label.move(20, 20)
        label.show()
        win.setLabel('left', 'avg losses',)
        win.setLabel('bottom', 'steps')
        curve1 =win.plot(pen = "r")
        curve2 =win.plot(pen = "b")
        curve1.setData(steps,avg_losses)
        curve2.setData(steps,avg_baseline_error)
        app.processEvents()
        try:
            while True:
                start_weight = weights["input_layer"]["node1"]["w2"]
                for ticker in data:
                    for i,date in enumerate(data[ticker]):
                        dates= list(data[ticker].keys())
                        if i != len(dates) - 1:
                            targetdate = dates[i+1]
                            target = data[ticker][targetdate]["percentage_change"]
                    
                        inputs = torch.tensor([
                            normalize(float(data[ticker][date]["Open"]),maximums[0],minimums[0]),               #open
                            normalize(float(data[ticker][date]["Close"]),maximums[1],minimums[1]),              #close
                            normalize(np.log10(float(data[ticker][date]["Volume"])),maximums[2],minimums[2]),   #volume
                            normalize(float(data[ticker][date]["High"]),maximums[3],minimums[3]),               #high
                            normalize(float(data[ticker][date]["Low"]),maximums[4],minimums[4]),                #low
                            float(data[ticker][date]["percentage_change"]),                                     #percentage change
                            normalize((data[ticker][date]["moving_average"]),maximums[5],minimums[5])           #5-day moving average
                        ])
                        final_output,nodes = input_layer(inputs,weights,nodes)
                        loss = (final_output-target)**2
                        loss.backward()
                        backpropagation(nodes,weights)
                        step =step+1
                        losses.append(float(loss))
                        bse = (target-0.00)**2 
                        baseline_error.append(float(bse))
                        if step >= 100: 
                            amount = sum(losses)
                            avg = (amount/len(losses))**0.5
                            avg_losses.append(avg)
                            amount = sum(baseline_error)
                            avg = (amount/len(baseline_error))**0.5
                            avg_baseline_error.append(avg)
                            steps.append(step)
                            losses.clear
                        if epoch > temp:
                            label.setText(f"epoch: {epoch:.6f}")
                            curve1.setData(steps,avg_losses)
                            curve2.setData(steps,avg_baseline_error)
                            app.processEvents()
                            temp = temp+1
                temp = epoch
                epoch = epoch+1            
                print(epoch)
                end_weight = weights["input_layer"]["node1"]["w2"]


        except KeyboardInterrupt:
            print("training paused")
            print("total epochs :",epoch)
            weights["epoch"] = epoch
            
            weights["graphdata"]["avg_losses"] = list(avg_losses)
            weights["graphdata"]["steps"] = list(steps)
            weights["graphdata"]["avg_baseline_error"] = list(avg_baseline_error)
            with open("weights.json","w") as file:
                json.dump(weights,file,indent = 4)

    if training == False:
        outputs = []
        choice = []
        with open(dataset,"r") as file:
            data = json.load(file)
            for ticker in data:
                for date in data[ticker]:
                    inputs = {
                        "op" : normalize(float(data[date][ticker]["Open"]),maximums[0],minimums[0]),                 #open
                        "cl" : normalize(float(data[date][ticker]["Close"]),maximums[1],minimums[1]),                #close
                        "vo" : normalize(np.log10(float(data[date][ticker]["Volume"])),maximums[2],minimums[2]),     #volume
                        "hi" : normalize(float(data[date][ticker]["High"]),maximums[3],minimums[3]),                 #high
                        "lo" : normalize(float(data[date][ticker]["Low"]),maximums[4],minimums[4]),                  #low
                        "pc" : float(data[date][ticker]["percentage_change"]),                                       #percentage change
                        "ma" : normalize((data[date][ticker]["moving_average"]),maximums[5],minimums[5])             #moving average
                    }
                    
                    final_output = input_layer(inputs)
                    outputs.append(final_output)
                    outputs = bubble_sort(outputs)
                    choice.append(outputs[0])
                    
            return choice       
        #switch normalisation to standardisation inputs to small making gradients tiny so extremeky little change
        
        