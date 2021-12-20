import pandas as pd
import plotly.express as px

from Features import Features
from Labels import Labels
from Models import Models

features = Features().test_features(data_points = 20)
print(features)
labels = Labels().test_labels_linear(features, noise = 0.01)
print(labels)
initial_model = Models([],regression = "linear")
initial_model.initialize_model(features)
initial_model.randomize_model()
print("starting_model:")
initial_model.print_model()