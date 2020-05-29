### 1
# In the first assignment we use keras library (https://keras.io/) to get familiar with TensorFlow.
# keras is a high-level API that is running on top of it.
# It offers a set of tools making working with TensorFlow easier.

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import pickle
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

from utils_victor import load_cifar_10_data, genr_bin_encodings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



### Ex. 1.1. Train a simply CNN on the CIFAR10 small images dataset.
# a. download the cifar10 data set from https://www.cs.toronto.edu/~kriz/cifar.html
# b. load and prepair the data: Convert it to numpy arrays. Divide it into the train and test sets. Normalize the images to the range [0,1]
...
# compare

cifar_10_dir = 'cifar-10-batches-py'
cifar_10_data = load_cifar_10_data(cifar_10_dir)
x_data_tr = cifar_10_data['train_data']
y_data_tr = cifar_10_data['train_labels']

x_data_te = cifar_10_data['test_data']
y_data_te = cifar_10_data['test_labels']
print('x_data_tr shape:', x_data_tr.shape) # expected: (50000, 32, 32, 3)
print('y_data_tr shape:', y_data_tr.shape) # expected: (50000,)
print('x_data_te shape:', x_data_te.shape) # expected: (10000, 32, 32, 3)
print('y_data_te shape:', y_data_te.shape) # expected: (10000,)
print('values > 1? :', np.any(x_data_tr>1), np.any(x_data_te>1)) # expected: False False

# c. convert class vectors to binary class matrices.
num_classes = 10
bin_class_matrix = genr_bin_encodings(num_classes)

# d. given the following simple CNN model, how many parameters has it?
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_data_tr.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# e. initiate the optimizer "opt", use RMSprop with parameters: lr=0.0001, decay=1e-6
opt = tf.keras.optimizers.RMSprop()

# f. compile the model, use parameters: loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# g. train the model, use "fit()" method of the Sequential model. Estimate acuracy on the test data after 10 and 25 epochs.
# During the training, go to the web page "https://keras.io/preprocessing/image/" and learn how to augment the data.
batch_size = 32
epochs = 25
model.fit(x_data_tr, bin_class_matrix[y_data_tr], epochs = 25, batch_size = 32)

# h. score and save the trained model.
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(x_data_te, bin_class_matrix[y_data_te],
                         batch_size=32)
print('test loss, test acc:', results)

# save the model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_trained_model.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


### Ex. 1.2. Finetune the pretrained CNN from the Ex. 1.1 with a proper data augmentation.
model.load_weights('.../cifar10_trained_model.h5')
epochs = 10
...
# score the trained model. Could you achieve a better accuracy cmp. to Ex. 1.1?
...
print('Test loss:', ...)
print('Test accuracy:', ...)


### Ex. 1.3. Take the pretrained network from the Ex. 1.1 and train it with dropout but without augmentation.
epochs = 10
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_data_tr.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.load_weights('.../cifar10_trained_model.h5')
...
# score the trained model. Could you achieve a better accuracy  cmp. to Ex. 1.1?
...
print('Test loss:', ...)
print('Test accuracy:', ...)


### Ex. 1.4. Now finetune the network with the augmentation and dropout
#  Estimate acuracy on the test data after 10 and 20 epochs. Could you achieve a better accuracy cmp. to Ex. 1.2 / 1.3?
epochs = 20
...
print('Test loss:', ...)
print('Test accuracy:', ...)
