"""
Lotto/Lottery Prediction 

Klasicni regresor 
MultiOutputRegressor
✅ SVR


svih 4504 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 31.10.2025.
"""


from qiskit_machine_learning.utils import algorithm_globals
import random

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
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
df = pd.read_csv("/data/loto7_4504_k86.csv", header=None)
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
Učitano kombinacija: 4504, Broj pozicija: 7
"""



# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df = df.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X = df.shift(1).dropna().values
y = df.iloc[1:].values

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)



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


# ✅ Train-test split mapiranih brojeva u indeksiranom opsegu
X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(X_x, y_x, test_size=0.25, random_state=39)


########################################

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)



X_train_scaled_x = scaler.fit_transform(X_train_x)
X_test_scaled_x = scaler.transform(X_test_x)

y_train_scaled_x = scaler.fit_transform(y_train_x)
y_test_scaled_x = scaler.transform(y_test_x)




# Create the SVR regressor
svr = SVR(epsilon=0.2)

# Create the Multioutput Regressor
mor = MultiOutputRegressor(svr)

# Train the regressor
mor1 = mor.fit(X_train, y_train)

# Train the regressor skaliran mapiran
mor2 = mor.fit(X_train_scaled_x, y_train_x)


# Generate predictions for testing data
y_pred1 = mor1.predict(X_test)
y_pred2 = mor2.predict(X_test_scaled_x)


predicted_numbers1 = mor1.predict(X_test[0].reshape(1, -1))
predicted_numbers2 = mor2.predict(X_test_scaled_x[0].reshape(1, -1))


predicted_numbers1 = np.round(predicted_numbers1).astype(int)
predicted_numbers2 = np.round(predicted_numbers2).astype(int)



print()
print("🎯 Predicted Next Lottery Numbers predicted_numbers1:", predicted_numbers1)
print("🎯 Predicted Next Lottery Numbers predicted_numbers2 skalirani mapirani:", predicted_numbers2)
print()
"""

🎯 Predicted Next Lottery Numbers predicted_numbers1: 
[[ 3  7 x x x  25 29]]
🎯 Predicted Next Lottery Numbers predicted_numbers2 skalirani mapirani: 
[[ 4  8 x x x 24 28]]

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
MSE for first regressor: 20.124630465739774 
     - second regressor: 32.95782536557552

MAE for first regressor: 3.0679824024813076 
     - second regressor: 4.370462017664307
"""




# Evaluate the regressor skaliran mapiran
mse_one_x = mean_squared_error(y_test_scaled_x[:,0], y_pred2[:,0])
mse_two_x = mean_squared_error(y_test_scaled_x[:,1], y_pred2[:,1])
print()
print(f'MSE_x for first regressor: {mse_one_x} - second regressor: {mse_two_x}')
print()

mae_one_x = mean_absolute_error(y_test_scaled_x[:,0], y_pred2[:,0])
mae_two_x = mean_absolute_error(y_test_scaled_x[:,1], y_pred2[:,1])
print()
print(f'MAE_x for first regressor: {mae_one_x} - second regressor: {mae_two_x}')
print()
"""
MSE_x for first regressor: 10.883350357890595 
       - second regressor: 55.51932325724881


MAE_x for first regressor: 3.123517484013979 
       - second regressor: 7.361783326327543
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
 [0.03703704 0.06666667 0.09677419 ... 0.48387097 0.82758621 0.85185185]
 [0.         0.03333333 0.25806452 ... 0.41935484 0.89655172 0.96296296]
 [0.         0.         0.09677419 ... 0.38709677 0.51724138 0.81481481]]
"""




X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_scale, y, test_size=0.25, random_state=39)

X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

# we now have a total of six variables 
print()
print("X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape")
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
print()
"""
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape
(3377, 7) (563, 7) (563, 7) (3377, 7) (563, 7) (563, 7)
"""



print()
print("\n✅ Script finished successfully.\n")
print()
"""
✅ Script finished successfully.
"""




"""
=== Qiskit Version Table ===
Software                       Version        
---------------------------------------------
qiskit                         1.4.4          
qiskit_machine_learning        0.8.3          

=== System Information ===
Python version                 3.11.13        
OS Apple                       Darwin 
"""
