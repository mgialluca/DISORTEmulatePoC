import numpy as np
import h5py
import keras
from keras import layers


# Load in training data 
train = h5py.File('/gscratch/vsm/gialluca/PostDocPropose/DISORT_Training_Data.h5', 'r')
x_train = np.array(train['IN'])
y_train = np.array(train['OUT'])


# Define Sequential model with 3 layers
# Much of the following code was modeled after various guides in keras.io
model = keras.Sequential(
    [
        layers.Dense(50, input_shape=(50,), activation="relu", name="input"),
        layers.Dense(128, activation="relu", name="layer2"),
        layers.Dense(256, activation="relu", name="layer3"),
        layers.Dense(200, name="output"),
    ]
)

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

model.fit(x_train, y_train, batch_size=1024, epochs=10)

# Evaluate on training data to see final mse / mae
results = model.evaluate(x_train, y_train)
print('evaluate results:')
print(results)

# Testing one input emulation to see output shape is correct 
predicttest = model.predict(x_train[:1])
print('Input shape: ', x_train[0].shape)
print('Prediction Shape: ', predicttest.shape)

# Save the model 
model.save('Version1.keras')