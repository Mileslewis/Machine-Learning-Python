class Neuron:
    def __init__(self, weights = [], layer = 0, activation="none",bInput = False,bOutput = False):
        self.weights = weights.copy()
        self.layer = layer
        self.activation = activation
        self.bInput = bInput
        self.bOutput = bOutput

    def print_neuron(self):
        print(self.weights)
        print(f"layer: {self.layer}")
        print("activation: " + self.activation)