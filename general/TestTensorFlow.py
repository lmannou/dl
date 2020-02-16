import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
 
# from keras.layers import Dropout # new!
# from keras.layers.normalization import BatchNormalization # new!
# from keras.optimizers import SGD
from matplotlib import pyplot as plt

# test tensorflow
# x=tf.random.normal(shape=(4,3))
# print("x",x)
# y=tf.random.normal(shape=(4,3))
# tf.print("y",y)
# z=tf.ones(shape=(1))
# tf.print("z", z)
# result = x*y + z
# tf.print("result", result)

# generate random date
import numpy as np
nb = 4
np.random.seed(42)
sizes = np.random.randint(low=1000, high=2000, size=nb)
tf.print(sizes)
np.random.seed(42)
prices = nb*100.0 + np.random.randint(low=1000, high=2000, size=nb)
tf.print(prices)

plt.plot(sizes, prices, "bx")
plt.ylabel("Price")
plt.xlabel("Sizes")
plt.show()

#load mnist
#mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # print(x_train[0])
# print(x_train.shape)

# plt.figure(figsize=(5,5))
# for k in range(12):
#  plt.subplot(3,4,k+1)
#  plt.imshow(x_train[k],cmap="Greys")
#  plt.axis('off')
# plt.tight_layout()
# plt.show()