import pandas as pd
import plotly.express as px

from Features import Features
from Labels import Labels
from Models import Models

features = Features().test_features(data_points = 5)
#print(features)
labels = Labels().test_labels_linear(features, noise = 0.005)
#print(labels)
initial_model = Models([],regression = "linear")
initial_model.initialize_model(features)
initial_model.add_layer(neurons = 2)
initial_model.randomize_model()
print("starting_model:")
initial_model.print_model()
total_losses = []
for i in range(10):
    total_losses.append(initial_model.update(features,labels,1,0.01))
print("final model:")
initial_model.print_model()
print(total_losses)
print(f"final average log loss: {initial_model.test(features,labels)}")
