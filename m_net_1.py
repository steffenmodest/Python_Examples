# -*- coding: utf-8 -*-
"""M-Net_1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/steffenmodest/Python_Examples/blob/Protocol/M_Net_1.ipynb
"""

import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

rs = 23  # fester Random Seed

# from google.colab import drive
# drive.mount('/content/drive')

"""Load Data"""

from tensorflow.keras.datasets import fashion_mnist, mnist

# Fashion MNIST Daten laden, wir wollen nur die Trainingsdaten und verwerfen die Testdaten
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""Own Funs"""

def ifelse(a):
  if a == 0:
    return 1
  else:
    return a


def rms_norm(v_x):
  n_rms = np.sqrt(np.sum(v_x ** 2)/(len(v_x) - 1))
  return v_x/n_rms


def prot_row(d_results):
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
    
    # colab file dir
    # prot_file = '/content/sample_data/M-Net_Protocol.xlsx'
    # local file dir
    prot_file = 'M-Net_Protocol.xlsx'

    if os.path.isfile(prot_file):
        df_prot = pd.read_excel (prot_file)
        df_prot = df_prot.append(d_results, 
                          ignore_index=True)
    else:
        # df_prot = pd.DataFrame.from_dict(d_results)
        df_prot = pd.DataFrame([results])
        # pf_prot = pd.DataFrame.from_records([d_results], index='TIMESTAMP')
        # df_prot = pd.DataFrame(columns=('TIMESTAMP', 
        #                                 'NUMBER_OF_HIDDEN', 
        #                                 'ACCURACY_SCORE', 
        #                                 'MAE_1ST_FL',
        #                                 'MSE_1ST_FL',
        #                                 'R2_1ST_FL',
        #                                 'MAE_1ST_BA',
        #                                 'MSE_1ST_BA',
        #                                 'R2_1ST_BA',
        #                                 'MAE_2ND_FL',
        #                                 'MSE_2ND_FL',
        #                                 'R2_2ND_FL',
        #                                 'MAE_2ND_BA',
        #                                 'MSE_2ND_BA',
        #                                 'R2_2ND_BA',
        #                                 'MSG'))
        
    # s_outcome = outcome.partition('<msg>')[2]
    # s_outcome = s_outcome.partition('</msg>')[0]
    # s_outcome = s_outcome.partition('Set ')[2]
    # s_set = s_outcome.partition(' ')[0]
    # s_msg = s_outcome.partition(' ')[2]
    
    # s_outcome = s_outcome.strip(',()')
    # l_outcome = s_outcome.split(',')
    # df_prot = df_prot.append({'TIMESTAMP':datetime.now(),
    #                           'RESULT': result,
    #                           'SAS_SET':int(s_set),
    #                           'MSG':s_msg,
    #                           'COUNT':n_len}, ignore_index=True)
    
    # df_prot = df_prot.append(d_results, 
    #                          ignore_index=True)
    df_prot.to_excel(prot_file, index=False)
        
    return None

"""Params and Results"""

params = {'N_HIDDEN': 680}
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

results

"""Preprocess data

NEW Norm
"""

X = X_train.reshape(len(X_train), -1).astype('int')
X = pd.DataFrame(X)
X = X.applymap(ifelse)
X = X.apply(rms_norm, axis=1)
X = np.array(X)

X

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Plätte 2D Bild in einen Vektor:
# Reshape behält die Anzahl Element in der ersten Dimension (len(X_orig) -> #Bilder)
# die restlichen Zellen des Arrays (-1 -> Pixel) bekommen alle ihre eigene Dimension
# X = X_train.reshape(len(X_train), -1).astype('float64')

# # Dimensionen um den Mittelpunkt zentrieren
# preproccessing = StandardScaler()
# X = preproccessing.fit_transform(X)
# X = np.array(X)

print ("Originaldaten:")
print("Shape: {}, Mean: {:f}, STD: {:f}".format(X_train.shape, np.mean(X_train), np.std(X_train)))

print ("Vorbereitete Daten:")
print("Shape: {}, Mean: {:f}, STD: {:f}".format(X.shape, np.mean(X), np.std(X)))

"""Create Hidden Activations"""

H = np.random.rand(60000, params['N_HIDDEN'])
print(H.shape)
print(H)

"""1st Forward Learn"""

import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(H, X)
print(reg.score(H, X))
X_pred = reg.predict(H)

results['MAE_1ST_FL'] = round(mean_absolute_error(X_pred, X), 3)
results['MSE_1ST_FL'] = round(mean_squared_error(X_pred, X), 3)
results['R2_1ST_FL'] = round(r2_score(X_pred, X, multioutput="variance_weighted"), 3)

print(f'MAE {mean_absolute_error(X_pred, X)}')
print(f'MSE {mean_squared_error(X_pred, X)}')
print(f'R2 {r2_score(X_pred, X)}')
print(f'R2 vw {r2_score(X_pred, X, multioutput="variance_weighted")}')
print(f'R2 ua {r2_score(X_pred, X, multioutput="uniform_average")}')

W = reg.coef_
print(W.shape)
WI = reg.intercept_
print(WI.shape)

"""1st Back Activations"""

# # X = X_train.reshape(len(X_train), -1).astype('int')
# W = pd.DataFrame(W)
# # X = X.applymap(ifelse)
# W = W.apply(rms_norm)
# W = np.array(X)

XT = X.transpose()
print(f'shape of X_train {X.shape} and X_train transposed {XT.shape}')

reg = LinearRegression().fit(W, XT)
print(reg.score(W, XT))
XT_pred = reg.predict(W)

results['MAE_1ST_BA'] = round(mean_absolute_error(XT_pred, XT), 3)
results['MSE_1ST_BA'] = round(mean_squared_error(XT_pred, XT), 3)
results['R2_1ST_BA'] = round(r2_score(XT_pred, XT, multioutput="variance_weighted"), 3)

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

# X = H_new_train.reshape(len(H_new_train), -1).astype('int')
H_new = pd.DataFrame(H_new)
# H_new = H_new.applymap(ifelse)
H_new = H_new.apply(rms_norm, axis=1)
H_new = np.array(H_new)

"""2nd Forward learn - result are Coherent Weights"""

reg = LinearRegression().fit(H_new, X)
print(reg.score(H_new, X))
X_pred = reg.predict(H_new)

results['MAE_2ND_FL'] = round(mean_absolute_error(X_pred, X), 3)
results['MSE_2ND_FL'] = round(mean_squared_error(X_pred, X), 3)
results['R2_2ND_FL'] = round(r2_score(X_pred, X, multioutput="variance_weighted"), 3)

print(f'MAE {mean_absolute_error(X_pred, X)}')
print(f'MSE {mean_squared_error(X_pred, X)}')
print(f'R2 {r2_score(X_pred, X)}')
print(f'R2 vw {r2_score(X_pred, X, multioutput="variance_weighted")}')
print(f'R2 ua {r2_score(X_pred, X, multioutput="uniform_average")}')

W = reg.coef_
print(W.shape)
WI = reg.intercept_
print(WI.shape)

"""2nd Back Activation - final representation of X_train in Hidden Activations"""

reg = LinearRegression().fit(W, XT)
print(reg.score(W, XT))
XT_pred = reg.predict(W)

results['MAE_2ND_BA'] = round(mean_absolute_error(XT_pred, XT), 3)
results['MSE_2ND_BA'] = round(mean_squared_error(XT_pred, XT), 3)
results['R2_2ND_BA'] = round(r2_score(XT_pred, XT, multioutput="variance_weighted"), 3)

print(f'MAE {mean_absolute_error(XT_pred, XT)}')
print(f'MSE {mean_squared_error(XT_pred, XT)}')
print(f'R2 vw {r2_score(XT_pred, XT, multioutput="variance_weighted")}')
print(f'R2 ua {r2_score(XT_pred, XT, multioutput="uniform_average")}')

H_new = reg.coef_
print(H_new.shape)
HI = reg.intercept_
print(HI.shape)

# X = H_new_train.reshape(len(H_new_train), -1).astype('int')
H_new = pd.DataFrame(H_new)
# H_new = H_new.applymap(ifelse)
H_new = H_new.apply(rms_norm, axis=1)
H_new = np.array(H_new)

"""KNN test with Hidden Activations"""

X = X_test.reshape(len(X_test), -1).astype('int')
X = pd.DataFrame(X)
X = X.applymap(ifelse)
X = X.apply(rms_norm, axis=1)
X = np.array(X)

# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]

# Plätte 2D Bild in einen Vektor:
# Reshape behält die Anzahl Element in der ersten Dimension (len(X_orig) -> #Bilder)
# die restlichen Zellen des Arrays (-1 -> Pixel) bekommen alle ihre eigene Dimension
# X_p_test = X_test.reshape(len(X_test), -1).astype('float64')

# # Dimensionen um den Mittelpunkt zentrieren
# preproccessing = StandardScaler()
# X_p_test = preproccessing.fit_transform(X_p_test)

print ("Originaldaten:")
print("Shape: {}, Mean: {:f}, STD: {:f}".format(X_test.shape, np.mean(X_test), np.std(X_test)))

print ("Vorbereitete Daten:")
print("Shape: {}, Mean: {:f}, STD: {:f}".format(X.shape, np.mean(X), np.std(X)))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')
neigh.fit(H_new, y_train)

# print(neigh.predict([[1.1]]))

# print(neigh.predict_proba([[0.9]]))

"""Back Activation of Test Data"""

XT = X.transpose()
print(f'shape of X_p_train {X.shape} and X_p_train transposed {XT.shape}')

reg = LinearRegression().fit(W, XT)
print(reg.score(W, XT))
XT_pred = reg.predict(W)

results['MAE_TEST_BA'] = round(mean_absolute_error(XT_pred, XT), 3)
results['MSE_TEST_BA'] = round(mean_squared_error(XT_pred, XT), 3)
results['R2_TEST_BA'] = round(r2_score(XT_pred, XT, multioutput="variance_weighted"), 3)

print(f'MAE {mean_absolute_error(XT_pred, XT)}')
print(f'MSE {mean_squared_error(XT_pred, XT)}')
print(f'R2 {r2_score(XT_pred, XT)}')
print(f'R2 vw {r2_score(XT_pred, XT, multioutput="variance_weighted")}')
print(f'R2 ua {r2_score(XT_pred, XT, multioutput="uniform_average")}')

H_test = reg.coef_
print(H_test.shape)
HI_test = reg.intercept_
print(HI_test.shape)

# X = H_new_train.reshape(len(H_new_train), -1).astype('int')
H_test = pd.DataFrame(H_test)
# H_new = H_new.applymap(ifelse)
H_test = H_test.apply(rms_norm, axis=1)
H_test = np.array(H_test)

# Predicting the Test set results
y_pred = neigh.predict(H_test)

# y_prob = neigh.predict_proba(H_test)

# print(y_prob[0])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score, classification_report

results['ACCURACY_SCORE'] = accuracy_score(y_test, y_pred)
results['DICT_REPORT'] = str(classification_report(y_test, y_pred, output_dict=True))
# results['DICT_REPORT'] = ''

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

prot_row(results)

pd.DataFrame([results])

results