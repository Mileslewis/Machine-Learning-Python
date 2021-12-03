import random

class Features:
    def __init__(self):
        self.features = []

    def test_features(self):
        TEST_DATA_POINTS = 40
        feature_const = []
        feature_x = []
        feature_y = []
        for x in range(TEST_DATA_POINTS):
            feature_const.append(1)  # for a constant term
            feature_x.append(random.random() * 6 - 3)
            feature_y.append(random.random() * 2 - 1)

        random.shuffle(feature_x)
        random.shuffle(feature_y)

        self.features.append(feature_const)
        self.features.append(feature_x)
        self.features.append(feature_y)
        return self.features
