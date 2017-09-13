# yazabi-tf-classification

This is an implementation of SharpestMinds' Tensorflow skill test. The train_and_validate function is in train.py, I also provided a convenient wrapper for calling it from the terminal.

__Examples of usage__:

./train.py logreg -i 4000

./train.py nn -i 6000 -H 256 -b 0.2

./train.py knn -K 5

The neural network model supports L2 regularization (through the -b parameter). I didn't implement it for logistic regression, since it's a fairly simple model and overfitting isn't really that much of a problem there. 
