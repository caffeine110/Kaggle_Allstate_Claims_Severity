#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:48:55 2018

@author: gaurav

"""


###############################################################################
### parameters tuning and optimisation
#fine-tuning
#grid search



###############################################################################
### importing data
def get_Data():
    ### importing data
    from pre import X_train, X_test, Y_train, Y_test
    return X_train, X_test, Y_train, Y_test



###############################################################################
### Building model
def build_Model(X_train, X_test, Y_train, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.advanced_activations import PReLU


    model = Sequential()
    
    model.add(Dense(output_dim = 1153, init = 'uniform', activation= 'relu', input_dim = X_train.shape[1]))

    model.add(Dense(output_dim = 832, init = 'uniform', activation='relu'))

    model.add(Dense(output_dim = 512, init = 'uniform', activation='relu'))

    model.add(Dense(output_dim = 256, init = 'uniform', activation='relu'))
    
    model.add(Dense(output_dim = 128, init = 'uniform', activation='relu'))

    model.add(Dense(output_dim = 64, init = 'uniform', activation='relu'))

    model.add(Dense(output_dim = 1, init = 'uniform', activation='relu'))
    
    ### Compile the network:
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model



###############################################################################
def load_Weights(Saved_Model):
    Saved_Model.load_weights('saved_Model/Saved_model_weights.h5')
    return Saved_Model


###############################################################################
def get_Predictions(Saved_Model, X_test):
    Y_pred = Saved_Model.predict(X_test)
    return Y_pred




###############################################################################
### Accuracy graph
def plot_Accuracy_Graph(Y_test, Y_pred):
    ### Predictions plot graph
    ### Accuracy graph
    y_original = Y_test[50:100]
    y_predicted = Y_pred[50:100]

    ### importing matplotlib
    import matplotlib.pyplot as plt
    
    plt.plot(y_original, 'r')
    plt.plot(y_predicted, 'b')
    
    plt.show()



###############################################################################
### Accure Scores
### Accure Scores
def accuracy_Score(Y_test, Y_pred):
    import sklearn.metrics
    res1 = sklearn.metrics.explained_variance_score(Y_test, Y_pred)
    print("Varience Score is : ",res1)





###############################################################################
def main():
    ### function call to get data
    X_train, X_test, Y_train, Y_test = get_Data()
    
    ### function call to build MOdel
    Saved_Model = build_Model(X_train, X_test, Y_train, Y_test)
    
    
    Saved_Model = load_Weights(Saved_Model)
    
    Y_pred = get_Predictions(Saved_Model, X_test)

    plot_Accuracy_Graph(Y_test, Y_pred)
    accuracy_Score(Y_test, Y_pred)

    pass



###############################################################################
if __name__ == '__main__':
    main()
