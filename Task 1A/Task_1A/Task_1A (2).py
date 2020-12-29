
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


def main():
    dir=os.path.join("files")
    data_train=pd.read_csv(
        os.path.join(dir,"train.csv"))
    x=data_train.iloc[:,2:]
    y=data_train.iloc[:,1]
    lambda_values=[0.01,0.1,1,10,100]
    results=[]
    n=10
    kf=KFold(n_splits=n)

  
    for i in range(0,5): 
        rmse=[]
        for train_index, test_index in kf.split(x):
            x_train = x.drop(test_index)
            x_test = x.drop(train_index)
            y_train = y.drop(test_index)
            y_train = y.drop(train_index)
            
            reg=Ridge(alpha=lambda_values[i])
            reg.fit(x_train, y_train)
            y_pred=reg.predict(x_test)

            rmse.append(sqrt(mean_squared_error(
                test_y,y_pred)))
        results.append(sum(rmse)/n)
  
    f=open("files/results.csv","w")
    for i in range(0,5):
        f.write(str(results[i]))
        f.write("\n")
    f.close()


    return






main()
