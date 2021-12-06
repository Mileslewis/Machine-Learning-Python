##  Uses an update model function to update the model once through all features and labels using a given batch size and learning rate.
#   Should work for any model/features/labels with the correct dimensions
#   Features are random points x, y within a certain range and a feature which is always 1 for a constant term
#   Labels are created from the target model which correspond to the created features and have an added noise amount
#   Initial model is created with random coefficients
#   Model is fed into update function which updates using gradient descent on the squared loss of each batch
#   Same initial model is semt through a given number of iterations with various different batch size/learning rate
#   Line graph of squared loss shows how the model converged for various batch sizes and learning rates
#   Models which reached squared loss over 1 million are frozen to keep graph clean + stop value errors
#   Optional l1 and l2 regularization included
#   Test function included to give loss for given features, labels and model.
import random
import pandas as pd
import plotly.express as px


############    update model function   ############

def update(model,features,labels,batch_size,learning_rate,l2=0,l1=0):
    i = 0
    total_loss = 0
    model_size = len(model)
    while i < len(labels):
        batch_end = i + batch_size
        if batch_end > len(labels):
            batch_end = len(labels)
        total_diff = [0 for j in range(model_size)]
        while i < batch_end:
            loss = 0
            for k in range(model_size):
                loss += model[k] * features[k][i]
            loss -= labels[i]
            #print(f"{i} loss: {loss}")
            total_loss += loss ** 2
            for j in range(model_size):
                total_diff[j] += 2 * features[j][i] * loss
        #print(f"diff: {total_diff}")
            i += 1
        for j in range(model_size):
            model[j] -= learning_rate * total_diff[j]
            if l2>0:
                model[j] -= 2 * l2 * model[j] * learning_rate
            if l1>0:
                if model[j]>0:
                    model[j] -= l1 * learning_rate
                    if model[j]<0:
                        model[j]=0
                elif model[j]<0:
                    model[j] += l1 * learning_rate
                    if model[j]>0:
                        model[j]=0

    return total_loss

def test(model,features,labels):
    total_loss = 0
    model_size = len(model)
    for i, label in enumerate(labels):
        loss = 0
        for k in range(model_size):
            loss += model[k] * features[k][i]
        loss -= label
        total_loss += loss ** 2
    return total_loss / max(1,len(labels))

##############    create randomised features     ###########
data_points = 40

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

#############   create labels with noise    ##########
noise = 0.01

target_model = [random.random() * 4 - 2, random.random() * 4 - 2, random.random() * 4 - 2]
print(f"Target Model: {target_model}")
labels = []


for i in range(data_points):
    label = 0
    for a in range(len(features)):
        label = label + target_model[a] * features[a][i] + 2 * random.random() * noise - noise
    labels.append(label)

initial_model = [random.random() * 4 - 2,random.random() * 4 - 2,random.random() * 4 - 2]
print(f"Initial Model: {initial_model}")

#############    set hyperparameters and make graphs   #########
learning_rate = [0.1, 0.05, 0.03, 0.01]
batch = [15,5,1]
repeats = 25
cutoff = 1000000

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
            total_losses.append(update(model,features,labels,B,L,l2=0.001,l1=0.001))
            #print(f"{iteration+1}: {model}")
            if total_losses[iteration] > cutoff:       # if the model has deconverged
                for i in range(iteration+1,repeats):
                    total_losses[iteration]= cutoff
                    total_losses.append(cutoff)
                break

            iteration = iteration + 1


        column = f"batch size: {B}, learning rate: {L}"
        df.insert(location, column, total_losses)
        location = location + 1
        columns.append(column)
        print(f"batch size: {B}, learning rate: {L}  final model: {model}, final average loss: {test(model,features,labels)}")

px.line(df,y=[x for x in columns], title = "Total Squared loss at each epoch for Initial Model with different Batch Sizes/Learning Rates", labels={'value': "squared loss", 'index': "epoch"}, log_y=True).show()


