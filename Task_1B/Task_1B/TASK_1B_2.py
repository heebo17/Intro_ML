import numpy as np
import pandas as pd
import matplotlib.pylab as ply

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV,
    cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge


def main():

    
    train_data = pd.read_csv('files/train.csv').drop('Id', axis=1)
    X_train = train_data.iloc[:, 1:].to_numpy()
    y_train = train_data['y'].to_numpy()

    pipeline = make_pipeline(FunctionTransformer(
                             funky_transform,
                             validate=True),
                         Ridge())
    parameter_space = {
        'ridge__alpha': np.logspace(-5, 5, 100)
    }
    inner_cv = KFold(n_splits=10, shuffle=True)
    outer_cv = KFold(n_splits=10, shuffle=True)
    classifier = GridSearchCV(pipeline, parameter_space,
                               n_jobs=1, scoring='neg_mean_squared_error',
                               iid=True,
                               refit=True,
                               cv=inner_cv)
    scores = cross_val_score(classifier, X_train, y_train.ravel(),
                            cv=outer_cv,
                            scoring='neg_mean_squared_error',
                            n_jobs=1)
    print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
          .format(score = np.sqrt(-scores).mean(), err = np.sqrt(-scores).std()))

    classifier = classifier.fit(X_train, y_train.ravel())
    coefficients = classifier.best_estimator_['ridge'].coef_

    
    df = pd.DataFrame(coefficients)
    df.to_csv('files/solution.csv', index=False, header=False)

    return

def funky_transform(X):
    linear = X
    quadratic = X ** 2
    exponential = np.exp(X)
    cosine = np.cos(X)
    constant = np.ones(X.shape[0])
    constant = constant[:, np.newaxis]
    return np.hstack((linear, quadratic, exponential, cosine, constant))


main()