# Machine-Learning-Python
Various machine learning files, written by me, but using knowledge from: https://developers.google.com/machine-learning/crash-course/ml-intro
These were mostly done to test out various concepts in Machine Learning (ML) but also to help me learn Python, since this is the first time I am using it.
I will try to give a brief desciption of what each file does here:

**Stocastic Linear:**
- My first file written.
- Creates some integer feature points within a certain range and corresponding labels = a + b * features.
- a and b are chosen constants. Random values are then chosen for w1 and w2 to create the model y = w1 + w2 * x 
- Gradient descent is used on the squared loss for each individual feature (Stochastic Gradient Descent as I understand it) to update the values of w1 and w2 in the model.
- 'learning_rate' and 'repeats' are also constants to be chosen. 'repeats' decides how many times the full set of features will be iterated through.
- A list is used to hold the average squared losses from the initial randomized model and from the model during each iteration. This is used for average squared loss graph later.
- A pandas dataframe is used to hold the features, labels and model predictions for each feature in the first randomized model and then after the end of each iteration through all the features
- A graph is then created from the dataframe to show the model's line graph at each of these times compared to the correct (labels) line graph.

**Stocastic Linear (better graphs):**
- Does the same as 'Stochastic Linear' but graphs are slightly different.
- Lines after each iteration are done with dashes and initial line is done with dots for clarity.
- Average squared losses show logarithmically.
- Probably won't be useful going forward but I thought the graphs looked better for this particular case.

Stochastic linear models did successfully converge to the correct line graphs so now we can try adding some complexity:
Next stage will be trying batch instead of stachastic gradient descent and also using more complicated models (more features).
Then Regularization, and separate training and test sets which are normalised and have noise included.
