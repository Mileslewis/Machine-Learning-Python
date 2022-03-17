import random
import pandas as pd

class Features:
    def __init__(self,data = None,names=[],offset = [],divisor = []):
        if data == None:
            self.data = []
        else:
            self.data = data
        self.names = names
        self.offset= offset
        self.divisor = divisor

    def test_features(self,const_feature = True,num_features = 2,data_points = 40,integer_only = False):
# Make a list of test feature lists, optional constant feature list (always 1) and optional rounding.
        if const_feature:
            feature_const = []
            #print(feature_const)
            for x in range(data_points):
                feature_const.append(1)  # for a constant term
            self.data.append(feature_const)
            self.names.append("c")
            self.offset.append(0)
            self.divisor.append(1)
        for i in range(num_features):
            feature = []
            for x in range(data_points):
                if integer_only:                   
                    feature.append(round(random.random() * 2 - 1))
                else:
                    feature.append((random.random() * 2 - 1))
            random.shuffle(feature)
            self.data.append(feature)
            self.names.append(str(i))
            self.offset.append(0)
            self.divisor.append(1)
        
        return self
