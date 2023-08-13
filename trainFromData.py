import tensorflow.keras as tf
from functions import *
import pickle

net = createNet([6, 6, 4, 2], 'relu')
#net = tf.models.load_model('net.h5')

# Load your data using pickle
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

# Separate inputs and outputs
inputs = [sample[0] for sample in data]
outputs = [sample[1] for sample in data]

# Convert inputs and outputs to NumPy arrays
import numpy as np
inputs = np.array(inputs)
outputs = np.array(outputs)/100

inputs = inputs.reshape(-1, 6)

batch_size = 200
num_epochs = 100

# Train your model
history = net.fit(inputs, outputs, batch_size=batch_size, epochs=num_epochs, verbose=1)
net.save('net6642.h5')