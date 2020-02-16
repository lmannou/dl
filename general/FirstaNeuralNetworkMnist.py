# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# First very simple Dense network, with no hidden layer.
# Start from here to improve the model in later iterations
# 2 layers:
# - Input: 64 units, sigmoid
# - Output: 10 units, softmax
# %% [markdown]
# Import packages

# %%
#numpy: Numeric library
import numpy as np 
#graphics
from matplotlib import pyplot as plt

#tensorflow: use some TF tools
import tensorflow as tf
#Keras: Deep learning API. In TF 2.x, it is included in TF
from tensorflow import keras
#Import Dense layer
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Model
#Optimizers: SGD (Stochastic Gradien Descent)
#from tensorflow.keras.optimizers import SGD as sgd

#import data set MNIST
from tensorflow.keras.datasets import mnist

# %% [markdown]
# Some verifications

# %%
#TF
print(np.random.uniform())

# %% [markdown]
# Import MNIST data set
# 60000 training examples
# 10000 validation examples
# 1 example = 28 * 28 matrix (image with 28 * 28 pixels)
# 

# %%
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()


# %%
print(x_train.shape)


# %%
print(y_train.shape)


# %%
print(x_train[0].shape)


# %%
print(y_train)
print(y_train.size)


# %%
print(x_valid.shape)
print(y_valid.shape)

# %% [markdown]
# Process & Prepare data
# Flattening Input data:
# Input data (x_train[i]) should be a vector of real values
# - From shape (60000, 28, 28) to shape (60000, 28 * 28)
# - Convert to float32 (default is uint8, which python will convert to float64) (flaot 32 will take less memory)

# %%
#print(x_train[0])


# %%
#x_train.reshape(60000, 784).astype('float32')


# %%
#use numpy reshape
x_train = np.reshape(x_train, (60000, 28 * 28)).astype('float32')
x_valid = np.reshape(x_valid, (10000, 28 * 28)).astype('float32')


# %%
print(x_train.shape)
print(x_valid.shape)

# %% [markdown]
# Prepare Data (2)
# divide by 255, so the values will range from 0 to 1
# 

# %%
#print(x_train)
x_train /=  255
x_valid /=  255


# %%
#print(x_train[0])

# %% [markdown]
# Prepare labels (y_train and y_valid)
# y_train[i] is given as a number (0, 9). We have 10 possible values. 
# Convert to one-hot format:
# 0 => [1, 0, 0, 0, 0,0, 0,0, 0,0]
# 1 => [0, 1, 0, 0, 0,0, 0,0, 0,0]
# 9 => [0, 0, 0, 0, 0,0, 0,0, 0,1]

# %%
n_classes = 10
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, n_classes)
print(y_train[0])

print(y_valid[0])
y_valid = keras.utils.to_categorical(y_valid, n_classes)
print(y_valid[0])

# %% [markdown]
# Define Neural Network Architecture
# Using Keras API

# %%
# Sequential: model of type sequential (layer n can pass information only to layer n+1)
model = Sequential()
model.to_json()


# %%
#Define activations
sigmoid = keras.activations.sigmoid
softmax = keras.activations.softmax

#define loss function
loss = keras.losses.mean_squared_error

#define metrics
metrics = [keras.metrics.Accuracy()]

#define optimizer
sgd=keras.optimizers.SGD()
print(sgd)


# %%
#define Input layer
inputSize = 28 * 28 # size of input vector
inputUnits = 64 #Number of input layer units (artificial neuron)
inputLayer = Dense(inputUnits, activation=  sigmoid, input_shape=(inputSize,))
model.add(inputLayer)


# %%
#define Output layer
 
outputUnits = 10 #Number of output layer units (artificial neuron)
# output shape will be inferred
outputLayer = Dense(outputUnits, activation=  softmax)
model.add(outputLayer)


# %%
#compile Model
model.compile(loss=loss, 
         optimizer=sgd,
         metrics=['accuracy'])


# %%
# print model summary
model.summary()


# %%
# Train
batch_size = 128
nb_epoch = 100
model.fit(x_train, y_train, batch_size= batch_size, epochs = nb_epoch, verbose=1, validation_data=(x_valid, y_valid))


# %%
# summary
model.summary()

# %% [markdown]
# With 100 epochs, this basic network seems pretty good, with minimal optimization, the accuracy is above 70%
