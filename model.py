#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:48:55 2018

@author: gaurav

"""

###############################################################################
### importing Data
def get_Data():
    from pre import X_train , X_test , Y_train , Y_test
    print(X_train.shape[1])

    return X_train , X_test , Y_train , Y_test


###############################################################################
### Building the Model
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

    model.add(Dense(output_dim = 1, init = 'uniform', activation='linear'))
    

    ### Compile the network:
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model	


###############################################################################
### printing the model summary
def get_Model_Summary(R_model):    
    R_model.summary()


###############################################################################
### Save model Weights
def save_Model(R_model):    
    SaveFileName = 'saved_Model/Saved_model_weights.h5'
    Saved_Model = R_model.save_weights(SaveFileName)
    type(R_model)


###############################################################################
### Make Predictions
def make_Predictions(Saved_model,X_test):
    Y_pred = Saved_model.predict(X_test)
    print(Y_pred)
    return Y_pred


# list all data in history
# summarize history for loss
def plot_Loss(History):
    ### importing matplotlib
    import matplotlib.pyplot as plt

    #dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

# summarize history for accuracy
def plot_Accuracy(History):
    ### importing matplotlib
    import matplotlib.pyplot as plt
    
    #dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    print(History.history.keys())
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()





#######################################################################################
### parameters tuning and optimisation
#fine-tuning
#grid search



def main():

    ### importing Data
    #import preprocessing
    X_train, X_test, Y_train, Y_test = get_Data()

    ### Building the Model
    R_model = build_Model(X_train, X_test, Y_train, Y_test)

    get_Model_Summary(R_model)
    
    ### Saving the checkpoints
    from keras.callbacks import ModelCheckpoint
    checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]

    ### training the model    
    History = R_model.fit(X_train, Y_train, epochs=1, batch_size=16, validation_split = 0.2, callbacks=callbacks_list)


    Y_pred = R_model.predict(X_test)
    print(Y_pred)
    print(Y_test)


    ### save the model
    save_Model(R_model)


    #plot_Accuracy(History)
    plot_Loss(History)

    print(type(History))
    print(History.history.keys())
    print(History.history.values())
    

    #Varience Score is :  0.5271425616083181
    print("Program Exicuted Succesfully")




if __name__ == "__main__":
    main()
    