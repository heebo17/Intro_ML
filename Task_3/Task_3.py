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
    print("read in:", data_train)

    #ENCODE THE DATA
    encoder = OneHotEncoder(sparse=False)
    x_train = encoder.fit_transform(feature_extractin(data_train))
    x_test = encoder.transform(feature_extractin(data_test))
    y_train = data_train["Active"].to_numpy()

    #Equit the data

    print("Class distribution: ", Counter(y_train.ravel()))
   


    datasets = create_datasets(x_train, y_train)

    print("Dataset:", datasets)

    for i, dataset in enumerate(datasets):
        print('Class distribution of dataset {}: '.format(i), Counter(dataset[1].ravel()))


    #Learning Pipeline

    for i, dataset in enumerate(datasets):
        print("Model {} cross-validation.".format(i))
        x_train_match = dataset[0]
        y_train_match = dataset[1]
        pipeline = make_pipeline(
            HistGradientBoostingClassifier(
                    learning_rate=0.21,
                    max_iter=200,
                    max_leaf_nodes=100,
                    min_samples_leaf=100)
        )
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        scores = cross_val_score(pipeline, x_train_match, y_train_match,
                            cv=cv,
                            scoring='f1',
                            n_jobs=-1,
                            verbose=True)
        print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
        .format(score = scores.mean(), err = scores.std()))
    
    #Fit on data
    estimators = []
    for i, dataset in enumerate(datasets):
        print("Model {} fitting to data.".format(i))
        x_train_match = dataset[0]
        y_train_match = dataset[1]
        estimators.append(HistGradientBoostingClassifier(
            learning_rate=0.21, max_iter = 200,
            max_leaf_nodes=100, min_samples_leaf=100))
        estimators[i].fit(x_train_match, y_train_match)

    ensemble_votes = VotingClassifier(estimators)

    preds = ensemble_votes.predict(x_test)
    df = pd.DataFrame(preds)
    df.to_csv("prediction.csv", index=False, header=False)


    Counter(y_train.ravel())
    Counter(preds.ravel())

    print("Class Distribution: ", Counter(y_train.ravel()))
    print("Prediction distribution: ", Counter(preds.ravel()))
    preid = ensemble_votes.predict(x_train)
    f1_score(y_train, preid)

    return


class VotingClassifier(object):

    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, x):
        Y=np.zeros([x.shape[0], len(self.estimators)], dtype=int)
        for i, clf in enumerate(self.estimators):
            Y[:, i] = clf.predict(x)
        y=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = np.argmax(np.bincount(Y[i,:]))
        return y.astype(np.int)


def create_datasets(X_train, y_train, ratio=100):
    datasets = []
    majority = (X_train[y_train == 0], y_train[y_train == 0])
    minority = (X_train[y_train == 1], y_train[y_train == 1])
    if minority[0].shape[0] * ratio > majority[0].shape[0]:
        return [(X_train, y_train)]
    for i in range(majority[0].shape[0] // (minority[0].shape[0] * ratio)):
        x_train, y_train = resample(*minority,
                                    replace=True, n_samples=(minority[0].shape[0] * ratio))
        x_train = np.concatenate([x_train, minority[0]], axis=0)
        y_train = np.concatenate([y_train, minority[1]], axis=0)
        datasets.append((x_train, y_train))
    return datasets

def feature_extractin(data):
    features = []
    for sequence in data["Sequence"].tolist():
        features.append([token for token in sequence])
    
    print(features)
    return features


main()