# Machine-Learning-Python
Various machine learning files, written by me, but using knowledge from: https://developers.google.com/machine-learning/crash-course/ml-intro
These were mostly done to test out various concepts in Machine Learning (ML) but also to help me learn Python, since this is the first time I am using it.
I will try to give a brief desciption of what each file does here:

## Model Classes
###### Features:
- Function to create test features with various options.

- Plan to have many functions for input data handling and a copy_format() function which can convert new data into the same features format used previously.

###### Labels:
- Functions to create test labels from inputed features.

- Again plan to have copy_format() function and other data handling functions.

###### Models:
- Models have a list of coefficiencts and then a 'regression' (linear or logistic) option.
- Function to create random model with 1 term for each feature.
- Copy() function to copy a model so that it can be updated multiple times from the same inital state to compare rsesults.
- update() function to update a model once through given features/labels set using batch gradient descent and regularization.
- test() function to test loss of model with given features/labels.

- Next will be increasing the complexity of model to allow neural network and potentially trying some different input methods.

###### main:
- Currently just calls the test functions which a model with each regression type with various hyperperamters and then creates graphs for comparisions.

- Hopefully will make some new scripts to try my classes out with real input data.


## Archive

###### Stocastic Linear:
- My first file written.
- Creates some integer feature points within a certain range and corresponding labels = a + b * features.
- a and b are chosen constants. Random values are then chosen for w1 and w2 to create the model y = w1 + w2 * x 
- Gradient descent is used on the squared loss for each individual feature (Stochastic Gradient Descent as I understand it) to update the values of w1 and w2 in the model.
- 'learning_rate' and 'repeats' are also constants to be chosen. 'repeats' decides how many times the full set of features will be iterated through.
- A list is used to hold the average squared losses from the initial randomized model and from the model during each iteration. This is used for average squared loss graph later.
- A pandas dataframe is used to hold the features, labels and model predictions for each feature in the first randomized model and then after the end of each iteration through all the features
- A graph is then created from the dataframe to show the model's line graph at each of these times compared to the correct (labels) line graph.

###### Stocastic Linear (better graphs):
- Does the same as 'Stochastic Linear' but graphs are slightly different.
- Lines after each iteration are done with dashes and initial line is done with dots for clarity.
- Average squared losses show logarithmically.
- Probably won't be useful going forward but I thought the graphs looked better for this particular case.

Stochastic linear models did successfully converge to the correct line graphs so now we can try adding some complexity:

###### Batch Regression:
- Uses Update model function which should be able to accept any well formed model+features+labels combination
- Does gradient descent on given batch sizes with given learning rate once through the whole set of labels
- Optional l1 and l2 regularization terms.
- test() function to test final model with a set of features and lables.
- Rest of the file is just generating Target model, Initial Model, features and labels to put into the update function
- Line graph generated with lines of squared loss for same Initial Model but with different batch sizes/ learning rates to compare

