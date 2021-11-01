## Does the same as 'Stochastic Linear' but graphs are slightly different.
# Lines after each iteration are done with dashes and initial line is done with dots for clarity.
# Average squared losses show logarithmically.
# Probably won't be useful going forward but I thought the graphs looked better for this particular case.

import random
import plotly.express as px
import plotly.graph_objects as go

learning_rate = 0.04
repeats = 10
a = 3
b = 2

def grad(w1, w2, feature, label):
    # Finds the gradient of the squared loss function for a function of form y = w1 + w2 * feature where 'label' is the correct result
    diff_w1 = 2 * (w1 + w2 * feature - label)
    diff_w2 = 2 * feature * (w1 + w2 * feature - label)
    return diff_w1, diff_w2

features = [x for x in range(-4, 5)]
labels = [a + b * x for x in features]


w1 = random.random() * 4 - 2
w2 = random.random() * 4 - 2
fig = go.Figure()   # graph of each model line, initially has dots, after each iteration has dashes, 'correct' line is normal line.
fig.add_trace(go.Scatter(x=features, y=[w1 + w2 * x for x in features], name="Initial",line=dict(width=4,dash='dot')))
#print(f"0: {w1} + {w2}x")

initial_losses = 0
for i, feature in enumerate(features):
    initial_losses = initial_losses + (w1 + w2 * features[i] - labels[i])**2
initial_losses = initial_losses / len(features)
losses = [initial_losses]       # list holds average squared loss initially and then during each iteration

ave_loss = 0
for j in range(repeats):
    for i, feature in enumerate(features):
        diff_w1, diff_w2 = grad(w1, w2, features[i], labels[i])
        #print(f"d{i + j * len(features)}: {(diff_w1,diff_w2)}")
        w1, w2 = w1 - diff_w1 * learning_rate, w2 - diff_w2 * learning_rate
        #print(f"{i+1 + j * len(features)}: {w1} + {w2}x")
        #print(f"loss = {(w1 + w2 * features[i] - labels[i])**2}")
        ave_loss = ave_loss + (w1 + w2 * features[i] - labels[i])**2
        if i == len(features) - 1:
            ave_loss = ave_loss/len(features)
            losses.append(ave_loss)
            ave_loss = 0
            fig.add_trace(go.Scatter(x=features, y=[w1 + w2 * x for x in features], name=j+1,line=dict(width=3,dash='dash')))
            
fig.add_trace(go.Scatter(x=features, y=labels, name="target",line=dict(width=4)))
px.line(losses, labels={'index': "iteration", 'value': "loss"}, title = "average squared loss during each Iteration through all points (Logarithmic)", log_y=True).show()
print(f"losses: {losses}")
print(f"final: {w1} + {w2}x")
fig.update_layout(title='line graphs after each iteration compared to target', xaxis_title='x', yaxis_title='y')
fig.show()