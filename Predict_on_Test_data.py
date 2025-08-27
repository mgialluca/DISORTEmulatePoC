import h5py
import keras 
import numpy as np
from datetime import datetime as dt

# Load in model
model = keras.models.load_model('Version1.keras')

hf = h5py.File('/gscratch/vsm/gialluca/PostDocPropose/DISORT_Testing_Data.h5', 'r')
x_test = np.array(hf['IN'])

# Determine how big the datasets should be:
inds = []
arr = np.array(hf['IN'])
for i in range(len(arr)-1):
    if arr[i][0] > arr[i+1][0]:
        inds.append(i+1)

# Run Prediction
st = dt.now() # to determine runtime
predictions = model.predict(x_test[:inds[0]])
np.save('Predictions_Test1.npy', predictions)
fn = dt.now()

print('Time for emulation 1: ')
print(fn-st)

st = dt.now() # to determine runtime
predictions = model.predict(x_test[inds[0]:inds[1]])
np.save('Predictions_Test2.npy', predictions)
fn = dt.now()

print('Time for emulation 2: ')
print(fn-st)

st = dt.now() # to determine runtime
predictions = model.predict(x_test[inds[1]:])
np.save('Predictions_Test3.npy', predictions)
fn = dt.now()

print('Time for emulation 3: ')
print(fn-st)


# Running on hyak for interactive node yielded runtimes of:
# 12.0 sec, 11.6 sec, 12.7 sec; avg of 12.1 sec

# Each DISORT run took 35 min 

# Emulation yielded over a 170x speed up
