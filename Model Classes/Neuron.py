import math

class Neuron:
    def __init__(self, weights = None, layer = 0, activation="none",bInput = False,bOutput = False, forwards_total = 0, backwards_gradient = 0):
        if weights == None:
            self.weights = []
        else:
            self.weights = weights
        self.layer = layer
        self.activation = activation
        self.bInput = bInput
        self.bOutput = bOutput
        self.forwards_total = forwards_total
        self.backwards_gradient = backwards_gradient
        self.weight_gradients = [0 for w in self.weights]

    def copy(self):
        return Neuron(weights = self.weights.copy(),layer = self.layer, activation=self.activation,bInput = self.bInput,bOutput = self.bOutput, forwards_total = 0, backwards_gradient = 0)

    def print_neuron(self):
        print("activation: " + self.activation)
        print(self.weights)

    def reset_gradients(self):
# weight gradients need to be reset after each batch has been finished and the model has been updated.
        self.weight_gradients = [0 for w in self.weights]

    def activate(self):
# modifies output value of neuron from it's input
        if self.activation == "none":
            return
        elif self.activation == "ReLU":
            if self.forwards_total > 0:
                return
            else:
                self.forwards_total = 0
                return
        elif self.activation == "sigmoid":
            self.forwards_total = 1 / (1 + math.exp(-self.forwards_total))
            return
        return
    
    def back_propagate(self):
# gives the differential of the model with respect to the input to this neuron.
        if self.activation == "none":
            return
        elif self.activation == "ReLU":
            if self.forwards_total > 0:
                return
            else:
                self.backwards_gradient = 0
                return
        elif self.activation == "sigmoid":
            self.backwards_gradient = self.backwards_gradient * (self.forwards_total) * (1 - self.forwards_total)
        return

