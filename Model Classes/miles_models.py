import random


class Models:
    def __init__(self, weights=[], regression="linear"):
        self.weights = weights
        self.regression = regression

    def copy(self):
        return Models(weights=self.weights.copy(), regression=self.regression)

    def print_weights(self):
        print(self.weights)

    def print_regression(self):
        print(self.regression)

    def random_model(self, features, size=1, norm=True):
        self.weights.clear()
        for i in range(len(features)):
            if norm:
                self.weights.append(size * 2 * random.random() - size)
            else:
                self.weights.append(size * random.random())

    def update(self, features, labels, batch_size, learning_rate, l2=0, l1=0):
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
                for k in range(model_size):
                    loss += self.weights[k] * features[k][i]
                loss -= labels[i]
                # print(f"{i} loss: {loss}")
                total_loss += loss ** 2
                for j in range(model_size):
                    total_diff[j] += 2 * features[j][i] * loss
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
        total_loss = 0
        model_size = len(self.weights)
        for i, label in enumerate(labels):
            loss = 0
            for k in range(model_size):
                loss += self.weights[k] * features[k][i]
            loss -= label
            total_loss += loss ** 2
        return total_loss / max(1, len(labels))
