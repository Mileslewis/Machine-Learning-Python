import pandas as pd
import numpy as np
import plotly.express as px

from Features import Features
from Labels import Labels
from Models import Models

    # ML takes time so turn off if unnecessary
ML = True 

    # import data
df = pd.read_csv('Data/QS_Dataset.csv') 

    # filter for high performers
m = df['last_evaluation'].mean()
df = df[df['last_evaluation']>m]    

    # normailize
for c in df.columns:
    if c != 'left'and c!= 'sales'and c!= 'salary'and c!= 'promotion_last_5years'and c!= 'Work_accident':
        df[c]= (df[c] - df[c].mean())/ df[c].std()

    # make salary integer high = 1, medium = 0, low = -1
df['salary']= (df['salary']=='high').astype(int)*2 + (df['salary']=='medium').astype(int)-1
#print(df['salary'])

    # find correlation
#print(df.corr())

    # create bool columns for each department
df["department"] = df["sales"]
df = df.drop(columns=['sales'])
if True:
    cats = df["department"].unique()
    #print(cats)
    for c in cats:
        df[c]=(df["department"]==c).astype(int)
df = df.drop(columns=['department'])

    # add constant column
df.insert(0,'constant',1)
#print(df.describe())  

    # split df into train and test (80 - 20)      
ar1,ar2,ar3,ar4,test = np.array_split(df.sample(frac = 1),5)
train = ar1.append(ar2).append(ar3).append(ar4)

    # 'left' column is labels
train_labels = train['left'].values.tolist()
train = train.drop(columns=['left'])

test_labels = test['left'].values.tolist()
test = test.drop(columns=['left'])

    # turn df into list for use as features.data
train_f = []
for c in train.columns:
    train_f.append(list(train[c].values))
#print(len(train_f))
#print(len(train_f[0]))

test_f = []
for c in test.columns:
    test_f.append(list(test[c].values))
#print(len(test_f))
#print(len(test_f[0]))

    # create features
train_features = Features(train_f)
test_features = Features(test_f)

    # create initial model with optional layers and regression options, then randomize weights
initial_model = Models(regression = "logistic")
initial_model.initialize_model(train_features,regression = "logistic")
#initial_model.add_layer(neurons = 5,activation = "sigmoid")
initial_model.randomize_model()
#print("starting_model:")
#initial_model.print_model()
#print("average_loss:")
#print(initial_model.test(features = test_features,labels = test_labels))

    # set hyperparameters. learning_rate and batch can be given as list to test many at once and plot on same graph
learning_rate = [0.008]
batch = [10]
repeats = 25
cutoff = 100000
l1 = 0.001 
l2 = 0.001

    # hold data for creating graph
df2 = pd.DataFrame()
columns = []
location = 0

if ML:
    for L in learning_rate:
        for B in batch:
    # copy initial model to test hyperparameters from same start
            model = initial_model.copy()
            # print(f"0: {model}")
            total_losses = []
            iteration = 0
            while iteration < repeats:
    # update model once through all features and record total loss for graph
                total_losses.append(
                    model.update(train_features, train_labels, B, L, l2, l1)
                )
                # print(f"{iteration+1}: {model}")
    # if the model has deconverged then cutoff
                if total_losses[iteration] > cutoff:  
                    for i in range(iteration + 1, repeats):
                        total_losses[iteration] = cutoff
                        total_losses.append(cutoff)
                    break

                iteration = iteration + 1
                #print("iteration : "+str(iteration))
                #model.print_model()
                #print("average_loss:")
                #print(model.test(features = train_features,labels = train_labels))

            column = f"batch size: {B}, learning rate: {L}"
            df2.insert(location, column, total_losses)
            location = location + 1
            columns.append(column)
        
            #print(f"batch size: {B}, learning rate: {L}, final average log loss: {model.test(features,labels)}")
    # print graph showing loss curves for each model
    px.line(
        df2,
        y=[x for x in columns],
        title="Total log loss at each epoch for Initial Model with different Batch Sizes/Learning Rates",
        labels={"value": "log loss", "index": "epoch"},
        log_y=True,
    ).show()

    # print and test final model (used once final hyperparameters are decided)
    print("final_model:")
    model.print_model()
    print("average_loss:")
    # model.test() returns: ave_loss, TP, FP, FN, TN
    print(model.test(features = test_features,labels = test_labels,return_confusion = True, threshold = 0.5))