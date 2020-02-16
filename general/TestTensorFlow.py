import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
 
# from keras.layers import Dropout # new!
# from keras.layers.normalization import BatchNormalization # new!
# from keras.optimizers import SGD
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train[0])
print(x_train.shape)

plt.figure(figsize=(5,5))
for k in range(12):
 plt.subplot(3,4,k+1)
 plt.imshow(x_train[k],cmap="Greys")
 plt.axis('off')
plt.tight_layout()
plt.show()