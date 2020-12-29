
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import matplotlib.pylab as plt
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_score)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

def main():

    #READ IN DATA
    data_train = pd.read_csv("files/train.csv")
    data_test = pd.read_csv("files/test.csv")

    
    x_train = data_train.drop("Active", axis=1)
    #ENCODE THE DATA
    encoder = OneHotEncoder(sparse=False)
    x_train = encoder.fit_transform(sequencing(data_train.drop("Active", axis=1)))
    x_test = encoder.transform(sequencing(data_test))
    y_train = data_train["Active"].to_numpy()
    print("Class distribution: ", Counter(y_train.ravel()))

    datasets = get_dataset(x_train, y_train)
    print(len(datasets))

   
    for i, dataset in enumerate(datasets):
        print("Class distribution: ", Counter(dataset[1].ravel()))
        x_train_up = dataset[0]
        y_train_up = dataset[1]

    #CREATING MODEL
    clf = HistGradientBoostingClassifier(
                learning_rate=0.2,
                max_iter=200,
                max_leaf_nodes=100,
                min_samples_leaf=100)


    

    clf.fit(x_train_up, y_train_up)
    preds = clf.predict(x_test)
    df = pd.DataFrame(preds)
    df.to_csv("predictions.csv", index= False, header=False)

    #EVALUATING MODEL

    pred_train = clf.predict(x_train)
    print("f1 score: ", f1_score(y_train, pred_train))
   
def get_dataset(X_train, y_train, ratio=8):
    datasets = []
    majority = (X_train[y_train == 0], y_train[y_train == 0])
    minority = (X_train[y_train == 1], y_train[y_train == 1])
    if minority[0].shape[0] * ratio > majority[0].shape[0]:
        return [(X_train, y_train)]
    for i in range(majority[0].shape[0] // (minority[0].shape[0] * ratio)):
        x_train, y_train = resample(*minority,
                                    replace=True, n_samples=(minority[0].shape[0] * ratio))
        x_train = np.concatenate([x_train, majority[0]], axis=0)
        y_train = np.concatenate([y_train, majority[1]], axis=0)
        datasets.append((x_train, y_train))
    return datasets
   
def sequencing(data):
    seq = []
    for sequence in data["Sequence"].tolist():
        seq.append([token for token in sequence])
    return seq

main()