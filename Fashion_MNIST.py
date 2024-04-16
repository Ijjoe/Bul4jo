import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

c4 = tf.constant([4])

print(c4)
#tensorflow                2.16.1  
print(c4.ndim)


# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# #print(train_images.shape, test_images.shape)
# #(60000, 28, 28) (10000, 28, 28)

# X = np.vstack((train_images, test_images))
# X = X.reshape([-1, 28*28])

# y=np.append(train_labels,test_labels)


# print('X.shape :', X.shape)
# print('y.shape :', y.shape)


# some_digit = X[50]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
#            interpolation="nearest")
# plt.axis("off")
# plt.show()
