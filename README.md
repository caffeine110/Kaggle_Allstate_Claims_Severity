
## AIM	: Kaggle_Allstate_Claims_Severity.
	: Predictive Analytics of Loss in Insurance


### Author	: caffeine110

### Introduction
Insurance Policy contains Data contains Numeric Pararmeters and Categorical parameters as a features
Insurance Loss is our Lable
By using DNN we have to find the Loss in Insurance for a perticular sample case.


### Keywords 
Keywords : Machine Learning, Data Analysis, Satastics, DNN, Numpy, Pandas.

## Tools
PreRequirements :
		 LIBRARIES	: Pandas,Numpy, Sklearn, Keras, TensorFlow, matplotlib, csv.
		 IDE		: spyder

###
Abstraact	: Using data manipulation libraries such as Numpy and Pandas for handle the huge dataset of Customers.
		  using matplot library we can visualise all the implimented modules.
		  Using sk-learn we can import test-train split method which divides the whole data into test and train cases.
		  Using keras we can build the DNN model with Sequential layers.
		  TensorFlow is an alternative library which allows to create ML model using Estemators and Tensors.


# procedure to run
Procedure : 

	1). Exctraction :
		This Problem Stagtemet is from a Kaggle Cometition
	2). Preporcessing
		Run the preprocessing.py or pre.py file to preprocess the downloaded data.
	3). Model Training
		Run the model.py file to fit data to model.
		While the model is trained program is under exicution and after complition apply the prediction steps.
	4). prediction
		To predict the Loss in Insurance put the data tupule in X_test cases.
		Output is shown in single Float Value which is a Loss in Insurance Policy.


# Evaluation Plan

As this is a Regression problem it is difficult to measure performance of Regressor model than the Classification One.
So we have mesure the performance of model using Varience Score and ploting the Graphs of expected and predicted values.


### key Metrics :


Variance—
	In terms of linear regression, variance Is a measure of how far observed
	values differ from the average of predicted values.
	Idely it is 1

Mean Absolute errorse(MAE)—
	It is a difference between two continues variables
	It is reduced from 2200 to 1255



# Optimisation :

### Parameter Tuning

We have tued the Parameters from the 

	Train-Test split from 60-40 ... 80-20 and get the best accuracy at 75-25

	Varing from 1... we choose Dense Layers : 6

	Number of Neurons in Layer each layers :
	We got best accuracy at layers at :
		: 1153 Neurons at input layer as Parameters are 1153
		: 832 Neurons at First Hidden layer
		: 512 Neurons at Second Hidden layer
		: 256 Neurons at Third Hidden layer
		: 128 Neuron at Fourth Hidden layer
		: 64 Neurons at Fifth Hidden layer
		: 1 Neuron at output layer as there is only 1 Output Loss in Insurance

	Tuned No of epoches 0 to 2 applied the Early_Stopping to stop model Training at 2
	Batch size 16
