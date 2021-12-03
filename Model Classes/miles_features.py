import random

class Features:
    def __init__(self):
        self.features = []

    def test_features(self,const_feature = True,num_features = 2,data_points = 40,integer_only = False):
        if const_feature:
            feature_const = []
            for x in range(data_points):
                feature_const.append(1)  # for a constant term
            self.features.append(feature_const)
        for c in range(num_features):
            feature = []
            for x in range(data_points):
                if integer_only:                   
                    feature.append(round(random.random() * 6 - 3))
                else:
                    feature.append((random.random() * 6 - 3))
            random.shuffle(feature)
            self.features.append(feature)
        
        return self.features
