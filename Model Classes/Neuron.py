class Neuron:
    def __init__(self, weights=[], activation="none",bInput = False,bOutput = False):
        self.weights = weights
        self.activation = activation
        self.bInput = bInput
        self.bOutput = bOutput

    def print_neuron(self):
        print(self.weights)
        print("activation: " + self.activation)