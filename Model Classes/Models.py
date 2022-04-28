import random
import math
from Neuron import Neuron

class Models:
    def __init__(self, neurons=None, regression="linear"):
        if neurons == None:
            self.neurons = []
        else:
            self.neurons = neurons
        self.regression = regression

    def copy(self):
        copied_neurons = []
        for j, layer in enumerate(self.neurons):
            copied_layer = []
            for k, neuron in enumerate(layer):
                copied_layer.append(neuron.copy())
            copied_neurons.append(copied_layer)
        return Models(neurons=copied_neurons, regression=self.regression)

    def print_model(self):
        for i,layer in enumerate(self.neurons):
            print(f"layer: {i}")
            for neuron in layer:
                neuron.print_neuron()

    def initialize_model(self, features, regression = "linear"):
# initialize a model with 1 input neuron for each feature and 1 output neuron.
        self.neurons.clear()
        input_neurons = []
        for i in range(len(features.data)):
            input_neurons.append(Neuron(bInput = True))
        self.neurons.append(input_neurons)
        self.neurons.append([Neuron(layer = 1, activation = regression,bOutput = True)])



    def randomize_model(self, size=1, norm=True):
# random starting weight for each neuron connection weight.
        for i in range(len(self.neurons)-1):
            #print(i)
            for j, neuron in enumerate(self.neurons[i]):
                for k in self.neurons[i+1]:
                    self.neurons[i][j].weights.append(random.random() * 2 * size - size)
                neuron.reset_gradients()

    def add_layer(self, neurons = 1, layer_pos = -1, activation = "none"):
# add layer of neurons at given position (default position is 1 layer before final (output) layer)
        if layer_pos == -1:
            layer_pos = len(self.neurons) - 1
        new_layer = []
        for i in range(neurons):
            new_layer.append(Neuron(layer = layer_pos, activation = activation))
        self.neurons.insert(layer_pos,new_layer)
        for layer in self.neurons[layer_pos+1:]:
            for neuron in layer:
                neuron.layer = neuron.layer + 1
        for neuron in self.neurons[layer_pos-1]:
            neuron.reset_gradients()
        


    def update(self, features, labels, batch_size, learning_rate, l2=0, l1=0):
# updates a model once through given features/labels set using batch gradient descent and regularization.
        i = 0
        total_loss = 0
        model_length = len(self.neurons)
        while i < len(labels):
            batch_end = i + batch_size
            if batch_end > len(labels):
                batch_end = len(labels)
            for layer in self.neurons:
                for neuron in layer:
                    neuron.reset_gradients()
            while i < batch_end:
                for layer in self.neurons:
                    for neuron in layer:
                        neuron.forwards_total = 0
                        neuron.backwards_gradient = 0
                loss = 0
                for j, layer in enumerate(self.neurons):
                    for k, neuron in enumerate(layer):
                        if neuron.bInput == True:
                            neuron.forwards_total = features.data[k][i]
                        if neuron.bOutput == False:
                            neuron.activate()
                            for l, weight in enumerate(neuron.weights):
                                self.neurons[j+1][l].forwards_total += neuron.forwards_total * weight
                        else:
                            if neuron.activation == "linear":
                                loss = neuron.forwards_total - labels[i]
                                #print(loss)
                                if loss < 10000:
                                    total_loss += loss ** 2
                                else:
                                     total_loss += 10000000
                                     return total_loss
                            elif neuron.activation == "logistic":
                                predicted = 1 / (1 + math.exp(max(-25,min(25,-neuron.forwards_total))))
                                if predicted > 0 and predicted < 1:
                                    loss = -labels[i] * math.log(predicted) - (1 - labels[i]) * math.log(1 - predicted)
                                    # print(f"{i} loss: {loss}")
                                else:
                                    print('Error ' + str(predicted) + ' ' + str(labels[i]) + ' ' + str(neuron.forwards_total))
                                    return total_loss + 10000000
                                total_loss += loss                             
                # print(f"{i} loss: {loss}")
                for j in range(model_length-1,-1,-1):
                    for k, neuron in enumerate(self.neurons[j]):
                        if neuron.bOutput == True:
                            if neuron.activation == "linear":
                                neuron.backwards_gradient = 2 * loss
                            elif neuron.activation == "logistic":
                                neuron.backwards_gradient = predicted - labels[i]
                        else:
                            for k, n in enumerate(self.neurons[j+1]):
                                grad = neuron.forwards_total * n.backwards_gradient
                                neuron.weight_gradients[k] += grad
                                neuron.backwards_gradient += grad
                                neuron.back_propagate()

                # print(f"diff: {total_diff}")
                i += 1
            for j, layer in enumerate(self.neurons):
                for k, neuron in enumerate(layer):
                    for l, weight in enumerate(neuron.weights):
                        #neuron.print_neuron()
                        #print(neuron.weight_gradients)
                        neuron.weights[l] -= learning_rate * neuron.weight_gradients[l]
                        if l2 > 0:
                            weight -= 2 * l2 * weight * learning_rate
                        if l1 > 0:
                            if weight > 0:
                                weight -= l1 * learning_rate
                                if weight < 0:
                                    weight = 0
                            elif weight < 0:
                                weight += l1 * learning_rate
                                if weight > 0:
                                    weight = 0

        return total_loss

    def test(self, features, labels, return_confusion = False, threshold = 0.5):
# tests loss of model with given features/labels.
        total_loss = 0
        if return_confusion == True:
            TP = 0
            FP = 0
            FN = 0
            TN = 0
        for i, label in enumerate(labels):
            for layer in self.neurons:
                for neuron in layer:
                    neuron.forwards_total = 0
                    neuron.backwards_gradient = 0           
            for j, layer in enumerate(self.neurons):
                for k, neuron in enumerate(layer):
                    if neuron.bInput == True:
                        neuron.forwards_total = features.data[k][i]
                    if neuron.bOutput == False:
                        for l, weight in enumerate(neuron.weights):
                            self.neurons[j+1][l].forwards_total += neuron.forwards_total * weight
                    else:
                        if neuron.activation == "linear":
                            loss = neuron.forwards_total - label
                            #print(loss)
                            total_loss += loss ** 2
                        elif neuron.activation == "logistic":
                            predicted = 1 / (1 + math.exp(-neuron.forwards_total))
                            #print(predicted)
                            #print(label)
                            loss = - label * math.log(predicted) - (1 - label) * math.log(1 - predicted)
                            #print(loss)
                            if return_confusion == True:
                                if predicted >= threshold:
                                    if label == 1:
                                        TP += 1
                                    else:
                                        FP +=1
                                else:
                                    if label == 1:
                                        FN += 1
                                    else:
                                        TN +=1                                    
                            total_loss += loss
        if return_confusion == True:
            return total_loss / max(1, len(labels)),TP,FP,FN,TN
        else:
            return total_loss / max(1, len(labels))
        
