import random

DATA_POINTS = 40


class Features:
    def __init__(self):
        self.feature_const = []
        self.feature_x = []
        self.feature_y = []
        self.features = []

    def main(self):
        for x in range(DATA_POINTS):
            self.feature_const.append(1)  # for a constant term
            self.feature_x.append(random.random() * 6 - 3)
            self.feature_y.append(random.random() * 2 - 1)

        random.shuffle(self.feature_x)
        random.shuffle(self.feature_y)

        self.features.append(self.feature_const)
        self.features.append(self.feature_x)
        self.features.append(self.feature_y)
        return self.features
