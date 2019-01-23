#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:56:25 2019

@author: gaurav
"""


################################################################################
### importing Data

#from preprocessing import main
#from preprocessing import X_train , X_test , Y_train , Y_test

from pre import X_train , X_test , Y_train , Y_test

print(X_train.shape[1])

#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test


###############################################################################
### Building the Model
def build_Model(X_train, X_test, Y_train, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.layers import BatchNormalization
    from keras.layers import Dropout
    #from keras.layers.advanced_activations import PReLU


    model = Sequential()
    model.add(Dense(351, input_dim=X_train.shape[1], init='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.578947))
    
    model.add(Dense(293, init='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.26666))
    
    model.add(Dense(46, init='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.188888))
    
    model.add(Dense(1, init='glorot_normal'))
    model.compile(loss='mae', optimizer='adadelta')
    return model

R_model = build_Model(X_train, X_test, Y_train, Y_test)


### print the summary
R_model.summary()









### Load the saved Weightserer
weights_file = 'checkpoints/Weights-002--1194.61794.hdf5' # choose the best checkpoint 

Saved_model = build_Model(X_train, X_test, Y_train, Y_test)
Saved_model.load_weights(weights_file) # load it
Saved_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

Y_pred = R_model.predict(X_test)

print(Y_pred)
print(Y_test)


#########################################################################################
### Predictions plot graph
### Accuracy graph
y_original = Y_test[50:100]
y_predicted = Y_pred[50:100]

### importing matplotlib
import matplotlib.pyplot as plt

plt.plot(y_original, 'r')
plt.plot(y_predicted, 'b')

plt.show()


#########################################################################################
### Accure Scores
import sklearn.metrics

### Calculating the Varience Score
res1 = sklearn.metrics.explained_variance_score(Y_test, Y_pred)
print("Varience Score is : ",res1)



#######################################################################################
### parameters tuning and optimisation
#fine-tuning
#grid search
