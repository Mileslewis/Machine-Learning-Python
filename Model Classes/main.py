import pandas as pd
import plotly.express as px

from miles_features import Features
from miles_labels import Labels
from miles_models import Models


def run():
    #############    set hyperparameters and make graphs   #########
    features = Features().test_features()
    labels = Labels().test_labels_linear(features)
    initial_model = Models([],regression = "linear")
    initial_model.random_model(features, 2)
    print("starting_model:")
    initial_model.print_weights()
    initial_model.print_regression()

    #############    set hyperparameters and make graphs   #########
    learning_rate = [0.1, 0.05, 0.03, 0.01]
    batch = [15, 5, 1]
    repeats = 50
    cutoff = 1000000

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
                    model.update(features, labels, B, L, l2=0.001, l1=0.001)
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
            print(f"batch size: {B}, learning rate: {L}  final model: {model.weights}, final average loss: {model.test(features,labels)}")
    px.line(
        df,
        y=[x for x in columns],
        title="Total Squared loss at each epoch for Initial Model with different Batch Sizes/Learning Rates",
        labels={"value": "squared loss", "index": "epoch"},
        log_y=True,
    ).show()


if __name__ == "__main__":
    run()
