# Author Bereket Kebede, Graduate Student
# Neural Networks -  Assignment #2 - University of Memphis. Fall 2021
# Question #1, Designing a Convolutional Neural Network (CNN)
# Last updated - Sept 23, 2021

#####################################################################################
# Import necessary libraries

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import BatchNormalization
import tensorflow as tf

#####################################################################################
# Load Training and Testing data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training Shape Data:")
print(X_train.shape)
print(y_train.shape)
print("Testing Shape Data:")
print(X_test.shape)
print(y_test.shape)

#####################################################################################
# Preview sample training image data

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Sample training images and its labels: ' + str([x[0] for x in y_train[0:5]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)

for j in range(5):
    img = X_train[j]
    axarr[j].imshow(img)
plt.show()

###################################################################################
# Load Training and Testing data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # Normalize pixel value of an image
X_test /= 255

print("Training Shape Data:")
print(X_train.shape)
print(y_train.shape)
print("Testing Shape Data:")
print(X_test.shape)
print(y_test.shape)

#####################################################################################
# Creating CNN Model for Image Classification

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=2, validation_split=0.2)
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
model.summary()

print(model.metrics_names)
print(score)

#####################################################################################
# Plotting training accuracy

print('Plotting training accuracy\n')
plt.plot(history.history['accuracy'])
plt.title('Classification Accuracy')
plt.ylabel('Accuracy')
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#####################################################################################
# Plotting training loss

print('Plotting training loss\n')
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

