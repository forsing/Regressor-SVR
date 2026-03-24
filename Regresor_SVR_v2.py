"""
Lotto/Lottery Prediction 

Klasicni regresor 
MultiOutputRegressor
✅ SVR


svih 4584 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 20.03.2026.
"""


from qiskit_machine_learning.utils import algorithm_globals
import random

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED


# ✅ Load data
df = pd.read_csv("/data/loto7_4584_k23.csv", header=None)
print()
print("✅ Data loaded successfully.")
print()
"""
✅ Data loaded successfully.
"""


print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4584, Broj pozicija: 7
"""



# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df = df.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X = df.shift(1).dropna().values
y = df.iloc[1:].values

# ✅ Train-test split (v2: vremenski — prvi 75% train, zadnjih 25% test)
_n = len(X)
_split = int(_n * 0.75)
X_train, X_test = X[:_split], X[_split:]
y_train, y_test = y[:_split], y[_split:]



####################################



# Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)


df_indexed = df_indexed.iloc[:, :7]


# Kreiranje ulaznih (X) i izlaznih (y) podataka
X_x = df_indexed.shift(1).dropna().values
y_x = df_indexed.iloc[1:].values


# ✅ Train-test split mapiranih brojeva u indeksiranom opsegu (v2: vremenski)
_nx = len(X_x)
_split_x = int(_nx * 0.75)
X_train_x, X_test_x = X_x[:_split_x], X_x[_split_x:]
y_train_x, y_test_x = y_x[:_split_x], y_x[_split_x:]


########################################

# Scale features (v2: samo ulaz X za mapirani grlo; odvojen scaler; y ostaje neskaliran u fit)
scaler_ix = StandardScaler()
X_train_scaled_x = scaler_ix.fit_transform(X_train_x)
X_test_scaled_x = scaler_ix.transform(X_test_x)

# Create the SVR regressor
# Create the Multioutput Regressor (v2: dva odvojena objekta — ne dva fit na istom mor)
mor_raw = MultiOutputRegressor(SVR(epsilon=0.2))
mor_ix = MultiOutputRegressor(SVR(epsilon=0.2))

# Train the regressor
mor_raw.fit(X_train, y_train)

# Train the regressor skaliran mapiran
mor_ix.fit(X_train_scaled_x, y_train_x)


# Generate predictions for testing data
y_pred1 = mor_raw.predict(X_test)
y_pred2 = mor_ix.predict(X_test_scaled_x)


_x_last = df.iloc[-1:].values.astype(float)
_x_last_ix = df_indexed.iloc[-1:].values.astype(float)
predicted_numbers1 = mor_raw.predict(_x_last)
predicted_numbers2 = mor_ix.predict(scaler_ix.transform(_x_last_ix))


predicted_numbers1 = np.round(predicted_numbers1).astype(int)
predicted_numbers2 = np.round(predicted_numbers2).astype(int)



print()
print("🎯 Predicted Next Lottery Numbers predicted_numbers1:", predicted_numbers1)
print("🎯 Predicted Next Lottery Numbers predicted_numbers2 skalirani mapirani:", predicted_numbers2)
print()
"""

🎯 Predicted Next Lottery Numbers predicted_numbers1: 
[[ 4  x 14 y 25 z 37]]
🎯 Predicted Next Lottery Numbers predicted_numbers2 skalirani mapirani: 
[[ 4  8 x 17 y 25 z]]

"""



# Evaluate the regressor
mse_one = mean_squared_error(y_test[:,0], y_pred1[:,0])
mse_two = mean_squared_error(y_test[:,1], y_pred1[:,1])
print()
print(f'MSE for first regressor: {mse_one} - second regressor: {mse_two}')
print()

mae_one = mean_absolute_error(y_test[:,0], y_pred1[:,0])
mae_two = mean_absolute_error(y_test[:,1], y_pred1[:,1])
print()
print(f'MAE for first regressor: {mae_one} - second regressor: {mae_two}')
print()
"""
MSE for first regressor: 17.123191526057976 - second regressor: 27.442626203057042


MAE for first regressor: 3.0150451246339247 - second regressor: 4.145395627298
"""




# Evaluate the regressor skaliran mapiran (v2: y u istom prostoru kao fit — neskaliran y_test_x)
mse_one_x = mean_squared_error(y_test_x[:,0], y_pred2[:,0])
mse_two_x = mean_squared_error(y_test_x[:,1], y_pred2[:,1])
print()
print(f'MSE_x for first regressor: {mse_one_x} - second regressor: {mse_two_x}')
print()

mae_one_x = mean_absolute_error(y_test_x[:,0], y_pred2[:,0])
mae_two_x = mean_absolute_error(y_test_x[:,1], y_pred2[:,1])
print()
print(f'MAE_x for first regressor: {mae_one_x} - second regressor: {mae_two_x}')
print()
"""
MSE_x for first regressor: 17.20820718836912 - second regressor: 27.63087537095055


MAE_x for first regressor: 3.0285272815983064 - second regressor: 4.157521058707108
"""



###################


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
print()
print("X_scale")
print(X_scale)
print()
"""
X_scale
[[0.14814815 0.4        0.38709677 ... 0.70967742 0.72413793 0.81481481]
 [0.03703704 0.03333333 0.32258065 ... 0.41935484 0.48275862 0.92592593]
 [0.44444444 0.5        0.48387097 ... 0.48387097 0.5862069  1.        ]
 ...
 [0.37037037 0.56666667 0.5483871  ... 0.58064516 0.93103448 0.96296296]
 [0.         0.1        0.25806452 ... 0.29032258 0.55172414 1.        ]
 [0.22222222 0.66666667 0.64516129 ... 0.80645161 0.86206897 0.96296296]]
"""




# v2: vremenska podela; druga imena da ne prepisuju train/test iznad
_nmm = len(X_scale)
_smm = int(_nmm * 0.75)
X_mm_train = X_scale[:_smm]
y_mm_train = y[:_smm]
X_mm_rest = X_scale[_smm:]
y_mm_rest = y[_smm:]
_smm2 = len(X_mm_rest) // 2
X_mm_val = X_mm_rest[:_smm2]
X_mm_test = X_mm_rest[_smm2:]
y_mm_val = y_mm_rest[:_smm2]
y_mm_test = y_mm_rest[_smm2:]

# we now have a total of six variables 
print()
print("X_mm_train.shape, X_mm_val.shape, X_mm_test.shape, y_mm_train.shape, y_mm_val.shape, y_mm_test.shape")
print(X_mm_train.shape, X_mm_val.shape, X_mm_test.shape, y_mm_train.shape, y_mm_val.shape, y_mm_test.shape)
print()
"""
X_mm_train.shape, X_mm_val.shape, X_mm_test.shape, y_mm_train.shape, y_mm_val.shape, y_mm_test.shape
(3437, 7) (573, 7) (573, 7) (3437, 7) (573, 7) (573, 7)
"""



print()
print("\n✅ Script finished successfully.\n")
print()
"""
✅ Script finished successfully.
"""
