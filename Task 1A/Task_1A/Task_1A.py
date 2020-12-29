import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt


def main():
    #Read in the data from train.csv and create a Dataframe
    #Seperating the Dataframe into x and y
    dir=os.path.join("files")
    data_train=pd.read_csv(
        os.path.join(dir,"train.csv"))
    x_train=data_train.iloc[:,2:]
    y_train=data_train.iloc[:,1]

    k=10
    lambdas=[0.1,1,10,100,1000]

    x_folds, y_folds = kfold(k, x_train, y_train)

    rmse_folds=rmse(lambdas, x_folds, y_folds, k)


    rmse_submit=pd.Series(rmse_folds)
    submit=rmse_submit.to_csv("files/sumbit.csv",index=False)
    
   
    return

def ridge(l, x_in, y_in, x_out, y_out):
    reg=Ridge(alpha=l)
    reg.fit(x_in, y_in)
    y_pred=reg.predict(x_out)

    return mean_squared_error(y_out, y_pred)

def kfold(k, x_train, y_train):
    x_folds=[]
    y_folds=[]
    kf=KFold(n_splits=k)
    for in_idx, out_idx in kf.split(x_train):
        x_folds.append((x_train.iloc[in_idx],x_train.iloc[out_idx]))
        y_folds.append((y_train.iloc[in_idx],y_train.iloc[out_idx]))
    return (x_folds, y_folds)

def rmse(lambdas, x_folds, y_folds, k):
    rmse_folds=[]
    for l in lambdas:
        rmse_avg=0
        for i in range(k):
            x_in, x_out=x_folds[i]
            y_in, y_out=y_folds[i]
            rmse=np.sqrt(ridge(l, x_in, y_in, x_out, y_out))
            rmse_avg += rmse
        rmse_avg/=k
        rmse_folds.append(rmse_avg)
    return rmse_folds


def main2():
    #Read in the data from train.csv and create a Dataframe
    #Seperating the Dataframe into x and y
    dir=os.path.join("files")
    data_train=pd.read_csv(
        os.path.join(dir,"train.csv"))
    x_train=data_train.iloc[:,2:]
    y_train=data_train.iloc[:,1]
    
    #Implementig lambda, results and KFold
    lambda_values=[0.01,0.1,1,10,100]
    results=[]
    n=10
    kf=KFold(n_splits=n)

    #First for loop iterates over all lambda values
    
    for i in range(0,5): 
        rmse=[]
        #Second for loop iterates over the 10 kfolds
    
        for train_index, test_index in kf.split(x):
           #Seperates the dataset into train and test

            train_x=x.drop(test_index)
            test_x=x.drop(train_index)
            train_y=y.drop(test_index)
            test_y=y.drop(train_index)
            
            #train the model with the train data set and the specific lambda value
            ridge_regressor=Ridge(alpha=lambda_values[i])
            ridge_regressor.fit(train_x,train_y)

            #predicts the y value on the test set of x
            y_pred=ridge_regressor.predict(test_x)

            #calculates the rmse value of the y predicted and the true y
            rmse.append(sqrt(mean_squared_error(
                test_y,y_pred)))
        #and in the end saves the rmse values of each fold in results list
        results.append(sum(rmse)/n)
  
    #Writes the results in a csv file

    f=open("files/results.csv","w")
    for i in range(0,5):
        f.write(str(results[i]))
        f.write("\n")
    f.close()


    return




main()
