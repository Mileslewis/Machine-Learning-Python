import pandas as pd
import plotly.express as px

from Features import Features
from Labels import Labels
from Models import Models

def test_linear():
    #############    set hyperparameters and make graphs   #########
    features = Features().test_features(data_points = 40)
    #print(features)
    labels = Labels().test_labels_linear(features, noise = 0)
    #print(labels)
    initial_model = Models([],regression = "linear")
    initial_model.initialize_model(features)
    initial_model.add_layer(neurons = 3)
    initial_model.randomize_model()
    print("starting_model:")
    initial_model.print_model()

    #############    set hyperparameters and make graphs   #########
    learning_rate = [0.1,0.05,0.03,0.01]
    batch = [1,5,15]
    repeats = 40
    cutoff = 1000

    df = pd.DataFrame()
    columns = []
    location = 0

    for L in learning_rate:
        for B in batch:
            model = initial_model.copy()
            # print(f"0: {model}")
            total_losses = []
            iteration = 0
            while iteration < repeats:
                total_losses.append(
                    model.update(features, labels, B, L, l2=0, l1=0)
                )
                # print(f"{iteration+1}: {model}")
                if total_losses[iteration] > cutoff:  # if the model has deconverged
                    for i in range(iteration + 1, repeats):
                        total_losses[iteration] = cutoff
                        total_losses.append(cutoff)
                    break

                iteration = iteration + 1

            column = f"batch size: {B}, learning rate: {L}"
            df.insert(location, column, total_losses)
            location = location + 1
            columns.append(column)
            #print(f"batch size: {B}, learning rate: {L}  final model: {model.weights}, final average loss: {model.test(features,labels)}")
    px.line(
        df,
        y=[x for x in columns],
        title="Total Squared loss at each epoch for Initial Model with different Batch Sizes/Learning Rates",
        labels={"value": "squared loss", "index": "epoch"},
        log_y=True,
    ).show()


def test_logistic():
    #############    set hyperparameters and make graphs   #########
    features = Features().test_features(data_points = 40,integer_only = True)
    #print(features)
    labels = Labels().test_labels_logistic(features)
    #print(labels)
    initial_model = Models([],regression = "logistic")
    initial_model.initialize_model(features,regression = "logistic")
    initial_model.add_layer(neurons = 3,activation = "ReLU")
    initial_model.randomize_model()
    print("starting_model:")
    initial_model.print_model()

    #############    set hyperparameters and make graphs   #########
    learning_rate = [0.1,0.05,0.03,0.01]
    batch = [1,5,15]
    repeats = 50
    cutoff = 1000

    df = pd.DataFrame()
    columns = []
    location = 0

    for L in learning_rate:
        for B in batch:
            model = initial_model.copy()
            # print(f"0: {model}")
            total_losses = []
            iteration = 0
            while iteration < repeats:
                total_losses.append(
                    model.update(features, labels, B, L, l2=0, l1=0)
                )
                # print(f"{iteration+1}: {model}")
                if total_losses[iteration] > cutoff:  # if the model has deconverged
                    for i in range(iteration + 1, repeats):
                        total_losses[iteration] = cutoff
                        total_losses.append(cutoff)
                    break

                iteration = iteration + 1

            column = f"batch size: {B}, learning rate: {L}"
            df.insert(location, column, total_losses)
            location = location + 1
            columns.append(column)
            #print(f"batch size: {B}, learning rate: {L}, final average log loss: {model.test(features,labels)}")
    px.line(
        df,
        y=[x for x in columns],
        title="Total log loss at each epoch for Initial Model with different Batch Sizes/Learning Rates",
        labels={"value": "log loss", "index": "epoch"},
        log_y=True,
    ).show()


if __name__ == "__main__":
    test_linear()
    test_logistic()
