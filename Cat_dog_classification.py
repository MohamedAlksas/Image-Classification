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

