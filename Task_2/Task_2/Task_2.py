import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics



def main():

    #READ IN DATA
    train_data = pd.read_csv("files/train_features.csv")
    test_data = pd.read_csv("files/test_features.csv")
    train_labels = pd.read_csv("files/train_labels.csv")

    
    #PREPROCESS DATA
    #FILLNAN VIA BACKPROP. ONLY FOR ONE PATIENT
    x_train = fillna(train_data, 12)
    x_test = fillna(test_data, 12)
    x_train = x_train.drop("Age", axis=1).drop("pid",axis=1).drop("Time",axis=1)
    x_test = x_test.drop("Age", axis=1).drop("pid",axis=1).drop("Time",axis=1)
    x_train = calc_time_data_all_labels(x_train.to_numpy(), 12)
    x_test = calc_time_data_all_labels(x_test.to_numpy(), 12)


    #DEFINE THE MODELS

    scaler = StandardScaler()
    clf = HistGradientBoostingClassifier(learning_rate=0.2)
    reg = HistGradientBoostingRegressor(learning_rate=0.2)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    #SUBTASK 1
    print("subtask1")
    
    df = pd.DataFrame({"pid": test_data.iloc[0::12,0].values})
    #SAVES ONLY LABEL RELEVANT DATA
    label_sub1_y_train = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                        'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 
                        'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                        'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

    y_train = train_labels[label_sub1_y_train].to_numpy()
    


    for i, label in enumerate(label_sub1_y_train):
        scores = cross_val_score(clf, x_train, y_train[:,i].ravel(),
                                 cv=5, verbose=True, scoring="roc_auc")
        print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))

    
    for i, label in enumerate(label_sub1_y_train):
        clf.fit(x_train, y_train[:,i].ravel())
        print("Training score:", 
            metrics.roc_auc_score(y_train[:,i], 
            clf.predict_proba(x_train)[:, 1]))
        predictions = clf.predict_proba(x_test)[:,1]
        df[label] = predictions


    #SUBTASK 2 
    print("subtask2")
    #FROM HERE NOT FROM ME


    subtask2_labels_ids = ['LABEL_Sepsis']
    y_train = train_labels[subtask2_labels_ids].to_numpy().ravel()

    scores = cross_val_score(clf, x_train, y_train,
                            cv=5,
                            scoring='roc_auc',
                            verbose=True)
    print("Cross-validation score is {score:.3f},"
      " standard deviation is {err:.3f}"
      .format(score = scores.mean(), err = scores.std()))

    
    clf.fit(x_train, y_train)
    predictions = clf.predict_proba(x_test)[:, 1]
    print("Training score:", metrics.roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1]))
    df[subtask2_labels_ids[0]] = predictions

    #Subtask 3

    print("subtask 3")

    
  
    subtask3_labels_ids = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2',
                      'LABEL_Heartrate']
    y_train = train_labels[subtask3_labels_ids].to_numpy()


    for i, label in enumerate(subtask3_labels_ids):
        scores = cross_val_score(reg, x_train, y_train[:, i],
                            cv=5,
                            scoring='r2',
                            verbose=True)
        print("Cross-validation score is {score:.3f},"
          " standard deviation is {err:.3f}"
          .format(score = scores.mean(), err = scores.std()))

    for i, label in enumerate(subtask3_labels_ids):
        reg.fit(x_train, y_train[:, i])
        predictions = reg.predict(x_test)
        print("Training score:", metrics.r2_score(y_train[:, i], reg.predict(x_train)))
        df[label] = predictions


    df.to_csv('prediction.csv', index=False, float_format='%.4f')
    
    return


def fillna(data, n_samples):
    x = []
    num_pat = int(data.shape[0]/n_samples)
    for index in range(num_pat):
        patient_data = data[n_samples * index:n_samples * (index + 1)]
        patient_data = patient_data.fillna(method="ffill")
        patient_data = patient_data.fillna(method="bfill")
        x.append(patient_data)
    df = pd.concat(x)
    df = df.fillna(df.mean())
    return df

def calc_time_data_all_labels(data, n_samples):
    y=[]
    for index in range(int(data.shape[0]/n_samples)):
        x=[]
        d = data[n_samples*index:n_samples * (index+1)]
        for i in range(12):
            x.append(d[i])
        df=np.concatenate(x, axis=0)
        c=df.T
        y.append(c)
    df2=np.stack(y, axis=0)
    return df2



 


main()

