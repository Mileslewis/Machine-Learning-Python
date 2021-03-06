import random

class Labels:
    def __init__(self,labels = None):
        if labels == None:
            self.labels = []
        else:
            self.labels = labels

    def test_labels_linear(self,features,noise = 0):
# makes a list of test labels from inputted features with a random 'target_model'
        target_model = [random.random() * 4 - 2 for a in features.data]
        print(f"Target Model: {target_model}")
        for i in range(len(features.data[0])):
            label = 0
            for a in range(len(features.data)):
                label += target_model[a] * features.data[a][i]
            label += 2 * random.random() * noise - noise
            self.labels.append(label)
        return self.labels

    def test_labels_logistic(self,features,noise = 0):
# makes a list of test labels from inputted features with a random 'target_model'
        target_model = [round(random.random() * 10 - 5) for a in features.data]
        print(f"Target Model: {target_model}")
        for i in range(len(features.data[0])):
            label = 0
            for a in range(len(features.data)):
                label += target_model[a] * features.data[a][i]
            label += 2 * random.random() * noise - noise
            if label>0:
                self.labels.append(1)
            else:
                self.labels.append(0) 
        return self.labels
