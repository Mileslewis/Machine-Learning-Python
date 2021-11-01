import random
import pandas as pd
import plotly.express as px

learning_rate = 0.04
repeats = 10
a = 3
b = 2

def grad(w1, w2, feature, label):
    # Finds the gradient of the squared loss function for a function of form y = w1 + w2 * feature where 'label' is the correct result
    diff_w1 = 2 * (w1 + w2 * feature - label)
    diff_w2 = 2 * feature * (w1 + w2 * feature - label)
    return diff_w1, diff_w2

features = range(-4, 5)
labels = [a + b * x for x in features]
df = pd.DataFrame(features,columns=['x'])   # dataframe to hold features, labels and model predictions for each iteration.

w1 = random.random() * 4 - 2
w2 = random.random() * 4 - 2
df.insert(loc=1,column = 0,value = [w1 + w2 * x for x in features])
#print(f"0: {w1} + {w2}x")
initial_losses = 0
for i, feature in enumerate(features):
    initial_losses = initial_losses + (w1 + w2 * features[i] - labels[i])**2
initial_losses = initial_losses / len(features)
losses = [initial_losses]       # list holds average squared loss initially and then during each iteration
columns = [0]       # used to hold the names of each column of 'df' for used in creating the line graph

ave_loss = 0
for j in range(repeats):
    for i, feature in enumerate(features):
        #print(f"loss = {(w1 + w2 * features[i] - labels[i])**2}")
        ave_loss = ave_loss + (w1 + w2 * features[i] - labels[i])**2
        diff_w1, diff_w2 = grad(w1, w2, features[i], labels[i])
        #print(f"d{i + j * len(features)}: {(diff_w1,diff_w2)}")
        w1, w2 = w1 - diff_w1 * learning_rate, w2 - diff_w2 * learning_rate
        #print(f"model {i+1 + j * len(features)}: {w1} + {w2}x")

        if i == len(features) - 1:
            ave_loss = ave_loss / len(features)
            losses.append(ave_loss)
            ave_loss = 0
            df.insert(loc=j+2,column = j+1,value = [w1 + w2 * x for x in features])
            columns.append(j+1)
df.insert(loc=repeats+2,column = "target",value = labels)
columns.append("target")
px.line(losses, labels={'index': "iteration", 'value': "loss"}, title = "average squared loss during each Iteration through all points").show()
print(f"losses: {losses}")
px.line(df,x='x',y=[x for x in columns],labels={'value': "y"}, title = "line graphs after each iteration compared to target").show()
print(f"final model: {w1} + {w2}x")



