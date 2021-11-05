
import random
import pandas as pd
import plotly.express as px

target_model = [0, 2, -2]
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
feature_const = []
feature_x = []
for x in range(-9,10):
    feature_const.append(1)
    feature_x.append(x)


feature_y = []
for y in range(-6,13):
    feature_y.append(y)

random.shuffle(feature_x)
random.shuffle(feature_x)
features = []
features.append(feature_const)
features.append(feature_x)
features.append(feature_y)
#print(features)


labels = []

for i in range(len(features[0])):
    label = 0
    for a in range(model_size):
        label = label + target_model[a] * features[a][i]
    labels.append(label)

initial_model = [2,-3,-5]

learning_rate = [0.009, 0.006, 0.003, 0.001]
batch = [5,3,1]
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
            iteration = iteration + 1
        column = f"batch size: {B}, learning rate: {L}"
        if total_losses[iteration - 1] < 1000:
            # if the model converged
            df.insert(location, column, total_losses)
            columns.append(column)
        print(f"batch size: {B}, learning rate: {L}  final model: {model}")

px.line(df,y=[x for x in columns], title = "Squared loss during each epoch for models which converged", labels={'value': "squared loss", 'index': "epoch"}, log_y=True).show()


