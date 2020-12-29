import numpy as np
import pandas as pd
import matplotlib.pylab as ply

from sklearn.model_selection import (GridSearchCV, cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main():

    #Read in Data from file and seperate into x and y
    train_data = pd.read_csv("files/train.csv").drop("Id", axis=1)
    x_train = train_data.iloc[:,1:].to_numpy()
    y_train = train_data["y"].to_numpy()

    #Add the missing data
    x2 = x_train**2
    x_exp = np.exp(x_train)
    x_cos = np.cos(x_train)
    x_ones = np.ones(x_train.shape[0])
    x_ones=x_ones[:,np.newaxis]
    x_train=np.hstack((x_train, x2, x_exp, x_cos, x_ones))

    
    
    
    #Reggression with Ridge
    alphas = 100
    reg=Ridge(alpha=alphas, fit_intercept=False)
    reg.fit(x_train,y_train)
    coeff=reg.coef_

    # Saves the results
    df=pd.DataFrame(coeff)
    df.to_csv("files/sample.csv", index=False, header=False)

    return



main()