import random
import math
from Neuron import Neuron

class Models:
    def __init__(self, neurons=[],weights = [], regression="linear"):
        self.neurons = neurons
        self.regression = regression
        self.weights = weights

    def copy(self):
        return Models(neurons=self.neurons.copy(), regression=self.regression)

    def print_model(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.print_neuron()

    def print_weights(self):
        print(self.neurons)

    def print_regression(self):
        print(self.regression)

    def initialize_model(self, features, regression = "linear"):
        self.neurons.clear()
        input_neurons = []
        for i in range(len(features)):
            input_neurons.append(Neuron(bInput = True))
        self.neurons.append(input_neurons)
        self.neurons.append([Neuron(activation = regression,bOutput = True)])



    def randomize_model(self, size=1, norm=True):
# random starting weight for each feature.
        for i in range(len(self.neurons)-1):
            print(i)
            for neuron in self.neurons[i]:
                for j in range(len(self.neurons[i+1])):
                    neuron.weights.append(random.random() * 2 * size - size)

    def update(self, features, labels, batch_size, learning_rate, l2=0, l1=0):
# updates a model once through given features/labels set using batch gradient descent and regularization.
        i = 0
        total_loss = 0
        model_size = len(self.weights)
        while i < len(labels):
            batch_end = i + batch_size
            if batch_end > len(labels):
                batch_end = len(labels)
            total_diff = [0 for j in range(model_size)]
            while i < batch_end:
                loss = 0
                if self.regression == "linear":
                    for k in range(model_size):
                        loss += self.weights[k] * features[k][i]
                    loss -= labels[i]
                    # print(f"{i} loss: {loss}")
                    total_loss += loss ** 2
                    for j in range(model_size):
                        total_diff[j] += 2 * features[j][i] * loss
                elif self.regression == "logistic":
                    value = 0
                    for k in range(model_size):
                        value += self.weights[k] * features[k][i]
                    predicted = 1 / (1 + math.exp(-value))
                    loss = - labels[i] * math.log(predicted) - (1 - labels[i]) * math.log(1 - predicted)
                    # print(f"{i} loss: {loss}")
                    total_loss += loss
                    for j in range(model_size):
                        total_diff[j] += features[j][i] * (predicted - labels[i])
                # print(f"diff: {total_diff}")
                i += 1
            for j in range(model_size):
                self.weights[j] -= learning_rate * total_diff[j]
                if l2 > 0:
                    self.weights[j] -= 2 * l2 * self.weights[j] * learning_rate
                if l1 > 0:
                    if self.weights[j] > 0:
                        self.weights[j] -= l1 * learning_rate
                        if self.weights[j] < 0:
                            self.weights[j] = 0
                    elif self.weights[j] < 0:
                        self.weights[j] += l1 * learning_rate
                        if self.weights[j] > 0:
                            self.weights[j] = 0

        return total_loss

    def test(self, features, labels):
# tests loss of model with given features/labels.
        total_loss = 0
        model_size = len(self.weights)
        for i, label in enumerate(labels):
            value = 0
            for k in range(model_size):
                value += self.weights[k] * features[k][i]
            if self.regression == "linear":
                loss = value - label
                #print(loss)
                total_loss += loss ** 2
            elif self.regression == "logistic":
                predicted = 1 / (1 + math.exp(-value))
                loss = - labels[i] * math.log(predicted) - (1 - labels[i]) * math.log(1 - predicted)
                total_loss += loss
        return total_loss / max(1, len(labels))
