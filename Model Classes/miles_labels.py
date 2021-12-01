import random

DATA_POINTS = 40


class Labels:
    def __init__(self, features=None) -> None:
        if not features:
            raise Exception("Labels need features")
        self.features = features
        self.noise = 0.01

        self.target_model = [
            random.random() * 4 - 2,
            random.random() * 4 - 2,
            random.random() * 4 - 2,
        ]
        print(f"Target Model: {self.target_model}")
        self.labels = []

    def main(self):
        for i in range(DATA_POINTS):
            label = 0
            for a in range(len(self.features)):
                label = (
                    label
                    + self.target_model[a] * self.features[a][i]
                    + 2 * random.random() * self.noise
                    - self.noise
                )
            self.labels.append(label)
        return self.labels
