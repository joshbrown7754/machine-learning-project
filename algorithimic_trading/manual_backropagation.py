import json
def backpropagation(loss_derivative,weights,node_outputs: dict):
    learning_rate = 0.01
    with open("weights.json","r") as file:
        weights = json.load(file)
        #outer layer calculations
        #bias
        weights["output_layer"]["bias"]["b4"] = weights["output_layer"]["bias"]["b4"] - (loss_derivative*learning_rate)
        #weights
        for i in range (1,len(weights["output_layer"]["node1"]+1)):#weights
            i = str(i)
            weight = weights["output_layer"]["node1"]["w"+i]
            weights["output_layer"]["node1"]["w"+i] =  weight- learning_rate*(loss_derivative*node_outputs["output_layer"][i])
            i = int(i)
        #layer3 calculations
        #bias
        for i in range (1,len(weights["layer3"]["bias"]+1)):
            i = str(i)
            bias = weights["layer3"]["bias"]["b"+i]
            input_out_relationship = weights["layer3"]["b"+i]
            if node_outputs["layer3"][i] != 0 :
                out_preRELU_relationship = 1
            else:
                out_preRELU_relationship = 0
            bias = bias - learning_rate*(loss_derivative*input_out_relationship*out_preRELU_relationship)
            weights["layer3"]["bias"]["b"+i] = bias
            i = int(i)
        #weights
        for i in range(len(weights["layer3"])):
            for  j in range (1,len(weights["layer3"]["node"+i])+1):
                j = str(j)
                weight  = weights["layer3"]["node"+j]["w"+j]
                weight_preRELU = node_outputs["layer2"][j]
                if node_outputs["layer3"][j] != 0 :
                    out_preRELU_relationship = 1
                else:
                    out_preRELU_relationship = 0
                input_out_relationship = weight
                weight = weight - learning_rate*(loss_derivative*input_out_relationship*out_preRELU_relationship*weight_preRELU)
                weights["layer3"]["node"+j]["w"+j] = weight
                j = int(j)
        