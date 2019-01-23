"""
Created on Sun Dec 30 11:48:55 2018

@author: gaurav

fieldNames = [ "id", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
              "cat10", "cat11", "cat12", "cat13", "cat14", "cat15", "cat16", "cat17", "cat18",
              "cat19", "cat20", "cat21", "cat22", "cat23", "cat24", "cat25", "cat26", "cat27",
              "cat28", "cat29", "cat30", "cat31", "cat32", "cat33", "cat34", "cat35", "cat36", 
              "cat37", "cat38", "cat39", "cat40", "cat41", "cat42", "cat43", "cat44", "cat45", 
              "cat46", "cat47", "cat48", "cat49", "cat50", "cat51", "cat52", "cat53", "cat54", 
              "cat55", "cat56", "cat57", "cat58", "cat59", "cat60", "cat61", "cat62", "cat63", 
              "cat64", "cat65", "cat66", "cat67", "cat68", "cat69", "cat70", "cat71", "cat72",
              "cat73", "cat74", "cat75", "cat76", "cat77", "cat78", "cat79", "cat80", "cat81", 
              "cat82", "cat83", "cat84", "cat85", "cat86", "cat87", "cat88", "cat89", "cat90",
              "cat91", "cat92", "cat93", "cat94", "cat95", "cat96", "cat97", "cat98", "cat99",
              "cat100", "cat101", "cat102", "cat103", "cat104", "cat105", "cat106", "cat107",
              "cat108", "cat109", "cat110", "cat111", "cat112", "cat113", "cat114", "cat115",
              "cat116", "cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7", "cont8",
              "cont9", "cont10", "cont11", "cont12", "cont13", "cont14", "loss"]


"""


import pandas as pd
import numpy as np

def read_files(filePath, fieldNames):
    
    df = pd.read_csv(filePath, usecols = fieldNames)

    return df


def parse_to_float(df):
    df = df.apply(pd.to_numeric)
    return df

def to_ndarrays(df):
    X = df.iloc[:,0:-1].values
    Y = df.iloc[:,14].values
    print(X)
    print(Y)
    
    return X,Y


def save_Files(X,Y):

    ### creating dataFrames to save intermediate data
    dataF_X = pd.DataFrame(X)
    dataF_Y = pd.DataFrame(Y)

    ### filenames to save data
    filePath_to_s_X = "datainter/encoded_X.csv"
    filePath_to_s_Y = "data/inter/encoded_Y.csv"

    ### writing files
    dataF_X.to_csv(filePath_to_s_X, index=False)
    dataF_Y.to_csv(filePath_to_s_Y, index=False)

    
def split_Data(X,Y):
    ### spli Data into 25%
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    return X_train, X_test, Y_train, Y_test



def scale_Features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


def return_Data(X_train, X_test, Y_train, Y_test):
    pass

def display_Metadata(df):
    #print(df.count())
    #print(df.count)
    print(df.dtypes)
    print(df.columns)
    print(df.head())
    print(df.info)
    

def display_Data(X_train, X_test, Y_train, Y_test):
    print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_test)




def main():

    ### filePath to open the files and read the data
    filePath = 'data/raw/train.csv'
    
    ### fieldNames to read Selected data Columns from data Files
    fieldNames = ["cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7", "cont8",
              "cont9", "cont10", "cont11", "cont12", "cont13", "cont14", "loss"]
    
    
    

    ### Function call to read files
    #df = read_files(filePath, fieldNames)

    df = pd.read_csv(filePath)
    cat_names = [c for c in df.columns if 'cat' in c]
    df = pd.get_dummies(data=df, columns=cat_names)
    
    df['Loss'] = df["loss"]

    df = df.drop("id",1)
    df = df.drop("loss",1)    

    print(df)
    
    ### convert to numeric
    df = parse_to_float(df)

    ### Function to display MetaData the Data about data
    #display_Metadata(df)
    
    ### converting into numpy arreys
    X,Y = to_ndarrays(df)
    
    
    ### Saving the intermediate data in files
    save_Files(X,Y)


    # Splitting the dataset into the Training set and Test set

    X_train, X_test, Y_train, Y_test = split_Data(X,Y)
    

    # Feature Scaling
    X_train, X_test = scale_Features(X_train, X_test)

    ### Display the Datasets
    display_Data(X_train, X_test, Y_train, Y_test)
    
    #return_Data( X_train, X_test, Y_train, Y_test)
    

if __name__ =="__main__":
    main()
