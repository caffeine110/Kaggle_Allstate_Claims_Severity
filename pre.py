### filePath to open the files and read the data
filePath = 'data/raw/train.csv'

### fieldNames to read Selected data Columns from data Files

fieldNames = ["cat107","cat108", "cat109", "cat110", "cat111", "cat112","cat113", "cat114",
              "cat115","cat116", 
              
              "cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7", "cont8",
              "cont9", "cont10", "cont11", "cont12", "cont13", "cont14", "loss"]


fieldNames = ["cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7", "cont8",
              "cont9", "cont10", "cont11", "cont12", "cont13", "cont14", "loss"]




### Function call to read files
#df = read_files(filePath, fieldNames)
import pandas as pd
import numpy as np

#import numpy as np
df = pd.read_csv(filePath)

cat_names = [c for c in df.columns if 'cat' in c]
df = pd.get_dummies(data=df, columns=cat_names)


#df['Loss'] = np.log(50 + df['loss'])
#df['Loss'] = np.antilog(df['loss'])

df['Loss'] = df['loss']

df = df.drop("id",1)
df = df.drop("loss",1)    

print(df.shape)
print(df.columns)

df = df.apply(pd.to_numeric)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values
print(X)
print(Y)

"""
df = pd.read_csv(filePath, usecols= fieldNames)

df['Loss'] = np.log(50 + df['loss'])
df = df.drop("loss",1)    

print(df.shape)

df = df.apply(pd.to_numeric)

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values
print(X)
print(Y)

### creating dataFrames to save intermediate data
dataF_X = pd.DataFrame(X)
dataF_Y = pd.DataFrame(Y)

### filenames to save data
filePath_to_s_X = "data/inter/encoded_X.csv"
filePath_to_s_Y = "data/inter/encoded_Y.csv"

### writing files
dataF_X.to_csv(filePath_to_s_X, index=False)
dataF_Y.to_csv(filePath_to_s_Y, index=False)
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[0:14] = sc.fit_transform(X[0:14])
#X_test = sc.fit_transform(X_test)


### split Data into 25%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)



