import h5py
import keras 
import numpy as np
from datetime import datetime as dt

# Load in model
model = keras.models.load_model('Version1.keras')

hf = h5py.File('/gscratch/vsm/gialluca/PostDocPropose/DISORT_Testing_Data.h5', 'r')
x_test = np.array(hf['IN'])

# Run Prediction
st = dt.now() # to determine runtime
# 3 datasets, each with 298620 wavelength points
predictions = model.predict(x_test[:298620])
np.save('Predictions_Test1.npy', predictions)
fn = dt.now()

print('Time for emulation 1: ')
print(fn-st)

st = dt.now() # to determine runtime
# 3 datasets, each with 298620 wavelength points
predictions = model.predict(x_test[298620:298620*2])
np.save('Predictions_Test2.npy', predictions)
fn = dt.now()

print('Time for emulation 2: ')
print(fn-st)

st = dt.now() # to determine runtime
# 3 datasets, each with 298620 wavelength points
predictions = model.predict(x_test[298620*2:298620*3])
np.save('Predictions_Test3.npy', predictions)
fn = dt.now()

print('Time for emulation 3: ')
print(fn-st)

