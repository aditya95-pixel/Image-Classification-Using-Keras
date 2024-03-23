from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import os
import numpy as np
img_width, img_height = 224, 224
train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 10
batch_size = 16
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
# Function to load and preprocess images
def load_and_preprocess_images(directory, target_size=(img_width, img_height)):
    images = []
    labels = []
    label_to_index = {}  # Dictionary to map label names to integer indices
    index = 0
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            label_to_index[label] = index
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = load_img(image_path, target_size=target_size)
                image = img_to_array(image) / 255.0  # Rescale to [0,1]
                images.append(image)
                labels.append(index)  # Assign integer label
            index += 1
    return np.array(images), np.array(labels)
# Load and preprocess training and validation images
X_train, y_train = load_and_preprocess_images(train_data_dir)
X_val, y_val = load_and_preprocess_images(validation_data_dir)
# Model architecture
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Fit the model
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)
model.save('model_saved.h5')
