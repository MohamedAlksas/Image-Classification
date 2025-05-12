import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
import os
from PIL import Image

# Function to remove corrupted images from a folder
def remove_corrupted_images(folder_path):
    removed = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Check if image is readable
            except Exception as e:
                print(f"Corrupted image removed: {filepath}")
                os.remove(filepath)
                removed += 1
    print(f"\nTotal corrupted images removed: {removed}")

# Remove corrupted images from training and testing datasets
remove_corrupted_images(r"C:\Users\momag\OneDrive\Desktop\cd_dataset\training_set")
remove_corrupted_images(r"C:\Users\momag\OneDrive\Desktop\cd_dataset\testing_set")
# Data augmentation and preprocessing for training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

training_set = train_datagen.flow_from_directory(r"C:\Users\momag\OneDrive\Desktop\cd_dataset\training_set",
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

# Preprocessing for test set (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(r"C:\Users\momag\OneDrive\Desktop\cd_dataset\testing_set",
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
                                            # Building the CNN model
cnn = tf.keras.models.Sequential()

# First convolution and pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second convolution and pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully connected output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing for training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(r"C:\Users\momag\OneDrive\Desktop\cd_dataset\training_set",
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Preprocessing for test set (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(r"C:\Users\momag\OneDrive\Desktop\cd_dataset\testing_set",
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


import random
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to test dataset
test_dir = r"C:\Users\momag\OneDrive\Desktop\cd_dataset\testing_set"

# Select a random class (e.g., 'cats' or 'dogs')
random_class = random.choice(os.listdir(test_dir))

# Select a random image from the chosen class
class_path = os.path.join(test_dir, random_class)
random_image_file = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, random_image_file)

# Load and preprocess the image for prediction
test_image = image.load_img(image_path, target_size=(64, 64))
img_array = image.img_to_array(test_image)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image

# Perform prediction
result = cnn.predict(img_array)

# Determine predicted label
predicted_label = 'Dog' if result[0][0] > 0.5 else 'Cat'

# Display the image with prediction
plt.imshow(test_image)
plt.title(f"Prediction: {predicted_label}")
plt.axis('off')
plt.show()