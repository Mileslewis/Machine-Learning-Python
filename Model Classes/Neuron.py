import math

class Neuron:
    def __init__(self, weights = [], layer = 0, activation="none",bInput = False,bOutput = False, forwards_total = 0, backwards_gradient = 0):
        self.weights = weights.copy()
        self.layer = layer
        self.activation = activation
        self.bInput = bInput
        self.bOutput = bOutput
        self.forwards_total = forwards_total
        self.backwards_gradient = backwards_gradient
        self.weight_gradients = [0 for w in weights]

    def copy(self):
        return Neuron(weights = self.weights.copy(),layer = self.layer, activation=self.activation,bInput = self.bInput,bOutput = self.bOutput, forwards_total = 0, backwards_gradient = 0)

    def print_neuron(self):
        print(self.weights)
        print(f"layer: {self.layer}")
        print("activation: " + self.activation)

    def reset_gradients(self):
        self.weight_gradients = [0 for w in self.weights]

    def activate(self):
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

