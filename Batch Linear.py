
import random
import pandas as pd
import plotly.express as px

target_model = [random.random() * 4 - 2, random.random() * 4 - 2, random.random() * 4 - 2]
print(f"Target Model: {target_model}")
data_points = 40
noise = 0
model_size = len(target_model)

def update(model,features,labels,batch_size,learning_rate):
    i = 0
    total_loss = 0
    while i < len(labels):
        batch_end = i + batch_size
        if batch_end > len(labels):
            batch_end = len(labels)
        total_diff = [0 for j in range(model_size)]
        while i < batch_end:
            loss = 0
            for k in range(model_size):
                loss = loss + model[k] * features[k][i]
            loss = loss - labels[i]
            #print(f"{i} loss: {loss}")
            total_loss = total_loss + loss ** 2
            for j in range(model_size):
                total_diff[j] = total_diff[j] + 2 * features[j][i] * loss
        #print(f"diff: {total_diff}")
            i = i + 1
        for j in range(model_size):
            model[j] = model[j] - learning_rate * total_diff[j]
    return(total_loss)

## create features
feature_const = []
feature_x = []
feature_y = []

for x in range(data_points):
    feature_const.append(1)     # for a constant term
    feature_x.append(random.random() * 6 - 3)
    feature_y.append(random.random() * 2 - 1)

random.shuffle(feature_x)
random.shuffle(feature_y)

features = []
features.append(feature_const)
features.append(feature_x)
features.append(feature_y)
#print(features)

## create labels with noise
labels = []

for i in range(data_points):
    label = 0
    for a in range(model_size):
        label = label + target_model[a] * features[a][i] + 2 * random.random() * noise - noise
    labels.append(label)

initial_model = [random.random() * 4 - 2,random.random() * 4 - 2,random.random() * 4 - 2]
print(f"Initial Model: {initial_model}")

## set hyperparameters
learning_rate = [0.1, 0.05, 0.03, 0.01]
batch = [15,5,1]
repeats = 40

df = pd.DataFrame()
columns = []
location = 0

for L in learning_rate:
    for B in batch:
        model = initial_model.copy()
        #print(f"0: {model}")
        total_losses = []
        iteration = 0
        while iteration < repeats:
            total_losses.append(update(model,features,labels,B,L))
            #print(f"{iteration+1}: {model}")
            if total_losses[iteration] > 1000000:       # if the model has deconverged
                iteration = iteration + 1
                break
            iteration = iteration + 1

        if total_losses[iteration - 1] <= 1000000:
            column = f"batch size: {B}, learning rate: {L}"
            df.insert(location, column, total_losses)
            location = location + 1
            columns.append(column)
        #print(f"batch size: {B}, learning rate: {L}  final model: {model}, final loss: {total_losses[iteration - 1]}")

px.line(df,y=[x for x in columns], title = "Squared loss during each epoch for models which converged", labels={'value': "squared loss", 'index': "epoch"}, log_y=True).show()


