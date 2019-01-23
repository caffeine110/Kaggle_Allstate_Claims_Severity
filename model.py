###############################################################################
### importing Data
def get_Data():
    #import preprocessing
    #from preprocessing import main
    #from preprocessing import X_train , X_test , Y_train , Y_test
    #print(X_train)
    #print(X_test)
    #print(Y_train)
    #print(Y_test
    
    from pre import X_train , X_test , Y_train , Y_test
    print(X_train.shape[1])
    return X_train , X_test , Y_train , Y_test




###############################################################################
### Building the Model
def build_Model(X_train, X_test, Y_train, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.layers import BatchNormalization
    from keras.layers import Dropout
    from keras.layers.advanced_activations import PReLU


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



###############################################################################
### printing the model summary
def get_Model_Summary(R_model):    
    ### print the summary
    R_model.summary()



###############################################################################
### Save model Weights
def save_Model(R_model):    

    ### Save model Weights
    SaveFileName = 'saved_Model/Saved_model_weights.h5'
    R_model.save_weights(SaveFileName)
    type(R_model)



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
    plt.ylabel('accuracy')
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

    
    ### Saving the checkpoints
    from keras.callbacks import ModelCheckpoint
    checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]
            
    ### training the model    
    History = R_model.fit(X_train, Y_train, epochs=2, batch_size=16, validation_split = 0.2, callbacks=callbacks_list)


    ### save the model
    save_Model(R_model)


    plot_Accuracy(History)
    plot_Loss(History)

    print(type(History))
    print(History.history.keys())
    print(History.history.values())
    


    print("Program Exicuted Succesfully")




if __name__ == "__main__":
    main()
    