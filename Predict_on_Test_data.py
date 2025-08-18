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
predictions = model.predict(x_test)
np.save('Predictions_version1.npy', predictions)
fn = dt.now()

print('Time for emulation: ')
print(fn-st)