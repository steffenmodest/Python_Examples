# -*- coding: utf-8 -*-
"""M-Net_1
Main file for Mnet
Has all funs
jupyter notebooks for execution in colab
"""

# from m_net_1 import H_test, X_pred
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow.keras.datasets import fashion_mnist, mnist

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, FunctionTransformer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# from google.colab import drive
# drive.mount('/content/drive')

def main(X, X_T, y_train, y_test, n_hidden=23):

    rs = 23  # fester Random Seed
    np.random.seed(rs)

    results, params = set_params_results(n_hidden)

    """Create Hidden Activations"""
    H = np.random.rand(60000, params['N_HIDDEN'])
    print(H.shape)
    print(H)

    """1st Forward Learn"""
    W, X_pred = forward_learn(H, X)
    results['MAE_1ST_FL'] = round(mean_absolute_error(X_pred, X), 3)
    results['MSE_1ST_FL'] = round(mean_squared_error(X_pred, X), 3)
    results['R2_1ST_FL'] = round(r2_score(X_pred, X, multioutput="variance_weighted"), 3)

    """1st Back Activations"""
    H, XT_pred = back_activation(W, X)
    XT = X.transpose()
    results['MAE_1ST_BA'] = round(mean_absolute_error(XT_pred, XT), 3)
    results['MSE_1ST_BA'] = round(mean_squared_error(XT_pred, XT), 3)
    results['R2_1ST_BA'] = round(r2_score(XT_pred, XT, multioutput="variance_weighted"), 3)

    """2nd Forward learn - result are Coherent Weights"""
    W, X_pred = forward_learn(H, X)
    results['MAE_2ND_FL'] = round(mean_absolute_error(X_pred, X), 3)
    results['MSE_2ND_FL'] = round(mean_squared_error(X_pred, X), 3)
    results['R2_2ND_FL'] = round(r2_score(X_pred, X, multioutput="variance_weighted"), 3)


    """2nd Back Activation - final representation of X_train in Hidden Activations"""
    H, XT_pred = back_activation(W, X)
    XT = X.transpose()
    results['MAE_2ND_BA'] = round(mean_absolute_error(XT_pred, XT), 3)
    results['MSE_2ND_BA'] = round(mean_squared_error(XT_pred, XT), 3)
    results['R2_2ND_BA'] = round(r2_score(XT_pred, XT, multioutput="variance_weighted"), 3)

    """Back Activation of Test Data"""
    H_test, XT_pred = back_activation(W, X_T)
    XT = X_T.transpose()
    results['MAE_TEST_BA'] = round(mean_absolute_error(XT_pred, XT), 3)
    results['MSE_TEST_BA'] = round(mean_squared_error(XT_pred, XT), 3)
    results['R2_TEST_BA'] = round(r2_score(XT_pred, XT, multioutput="variance_weighted"), 3)

    y_pred = knn_test(H_test, H, y_train)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    results['ACCURACY_SCORE'] = accuracy_score(y_test, y_pred)
    results['DICT_REPORT'] = str(classification_report(y_test, y_pred, output_dict=True))

    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    df_results = pd.DataFrame([results])
    prot_row(df_results)

    return

"""Own Funs"""

def preprocess(X_train, X_test):

    # NEW Norm

    X = X_train.reshape(len(X_train), -1).astype('int')
    X = pd.DataFrame(X)
    X = X.applymap(ifelse)
    X = X.apply(rms_norm, axis=1)
    X = np.array(X)

    print ("Originaldaten:")
    print("Shape: {}, Mean: {:f}, STD: {:f}".format(X_train.shape, np.mean(X_train), np.std(X_train)))

    print ("Vorbereitete Daten:")
    print("Shape: {}, Mean: {:f}, STD: {:f}".format(X.shape, np.mean(X), np.std(X)))

    X_T = X_test.reshape(len(X_test), -1).astype('int')
    X_T = pd.DataFrame(X_T)
    X_T = X_T.applymap(ifelse)
    X_T = X_T.apply(rms_norm, axis=1)
    X_T = np.array(X_T)

    print ("Originaldaten:")
    print("Shape: {}, Mean: {:f}, STD: {:f}".format(X_test.shape, np.mean(X_test), np.std(X_test)))

    print ("Vorbereitete Daten:")
    print("Shape: {}, Mean: {:f}, STD: {:f}".format(X_T.shape, np.mean(X_T), np.std(X_T)))

    return X, X_T

def ifelse(a):
    if a == 0:
        return 1
    else:
        return a


def rms_norm(v_x):
    n_rms = np.sqrt(np.sum(v_x ** 2)/(len(v_x) - 1))
    return v_x/n_rms


def prot_row(df_results):
    """
    make a new row in protokoll

    Parameters
    ----------
    outcome : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Colab file dir also on local 
    prot_file = '/content/sample_data/M-Net_Protocol.xlsx'

    # local file dir old
    # prot_file = 'M-Net_Protocol.xlsx'
    if os.path.isfile(prot_file):
        df_prot = pd.read_excel (prot_file)
        df_prot = df_prot.append(df_results, 
                          ignore_index=True)
    else:
        df_prot = df_results

    df_prot.to_excel(prot_file, index=False)
        
    return None

"""Params and Results"""
def set_params_results(n_hidden):

  params = {'N_HIDDEN': n_hidden}
  results = {'TIMESTAMP': datetime.now(),
            'NUMBER_OF_HIDDEN': params['N_HIDDEN'],
            'ACCURACY_SCORE': 0, 
            'MAE_1ST_FL': 0,
            'MSE_1ST_FL': 0,
            'R2_1ST_FL': 0,
            'MAE_1ST_BA': 0,
            'MSE_1ST_BA': 0,
            'R2_1ST_BA': 0,
            'MAE_2ND_FL': 0,
            'MSE_2ND_FL': 0,
            'R2_2ND_FL': 0,
            'MAE_2ND_BA': 0,
            'MSE_2ND_BA': 0,
            'R2_2ND_BA': 0,
            'MAE_TEST_BA': 0,
            'MSE_TEST_BA': 0,
            'R2_TEST_BA': 0,
            'DICT_REPORT': '',
            'MSG':'M-Net on python'}

  return(results, params)


def forward_learn(H, X):

    reg = LinearRegression().fit(H, X)
    print(reg.score(H, X))
    X_pred = reg.predict(H)

    print(f'MAE {mean_absolute_error(X_pred, X)}')
    print(f'MSE {mean_squared_error(X_pred, X)}')
    print(f'R2 {r2_score(X_pred, X)}')
    print(f'R2 vw {r2_score(X_pred, X, multioutput="variance_weighted")}')
    print(f'R2 ua {r2_score(X_pred, X, multioutput="uniform_average")}')

    W = reg.coef_
    print(W.shape)
    WI = reg.intercept_
    print(WI.shape)

    return W, X_pred


def back_activation(W, X):

    XT = X.transpose()
    print(f'shape of X_train {X.shape} and X_train transposed {XT.shape}')

    reg = LinearRegression().fit(W, XT)
    print(reg.score(W, XT))
    XT_pred = reg.predict(W)

    print(f'MAE {mean_absolute_error(XT_pred, XT)}')
    print(f'MSE {mean_squared_error(XT_pred, XT)}')
    print(f'R2 {r2_score(XT_pred, XT)}')
    print(f'R2 vw {r2_score(XT_pred, XT, multioutput="variance_weighted")}')
    print(f'R2 ua {r2_score(XT_pred, XT, multioutput="uniform_average")}')

    H_new = reg.coef_
    print(H_new.shape)
    HI = reg.intercept_
    print(HI.shape)

    """Hidden Norm"""
    H_new = pd.DataFrame(H_new)
    H_new = H_new.apply(rms_norm, axis=1)
    H_new = np.array(H_new)

    return H_new, XT_pred


"""KNN test with Hidden Activations"""

def multi_knn_test(H_test, H, y_train,):
    """ Majority of 3 kNN with different k - Weigthed kNN has similar results with one k """

    # k1
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')
    neigh.fit(H, y_train)
    # Predicting the Test set results
    y_pred_1 = neigh.predict(H_test)
    # k2
    neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, weights='distance')
    neigh.fit(H, y_train)
    # Predicting the Test set results
    y_pred_2 = neigh.predict(H_test)
    # k3
    neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, weights='distance')
    neigh.fit(H, y_train)
    # Predicting the Test set results
    y_pred_3 = neigh.predict(H_test)

    y_pred_all = pd.DataFrame(y_pred_1)
    y_pred_all['2'] = y_pred_2
    y_pred_all['3'] = y_pred_3
    y_pred_all['majority'] = y_pred_all.mode(axis=1)[0]

    y_pred = y_pred_all['majority'].to_numpy()

    return y_pred

def knn_test(H_test, H, y_train,):

    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')
    neigh.fit(H, y_train)
    # Predicting the Test set results
    y_pred = neigh.predict(H_test)

    return y_pred

def load_data():

    # Fashion MNIST Daten laden, wir wollen nur die Trainingsdaten und verwerfen die Testdaten
    # (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test


# Code starting point

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    X, X_T = preprocess(X_train, X_test)
    for i in range(7, 8, 1):
        main(X, X_T, y_train, y_test, n_hidden=i)
