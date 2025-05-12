# Import Important Library

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import random
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Remove Corrupted Images
def remove_corrupted_images(folder_path):
    removed = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  
            except Exception as e:
                print(f"Corrupted image removed: {filepath}")
                os.remove(filepath)
                removed += 1
    print(f"\nTotal corrupted images removed: {removed}")

remove_corrupted_images(r"C:\Users\momag\OneDrive\Desktop\Projects_for_ alGam3aa\cd_dataset\training_set")
remove_corrupted_images(r"C:\Users\momag\OneDrive\Desktop\Projects_for_ alGam3aa\cd_dataset\testing_set")

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(r"C:\Users\momag\OneDrive\Desktop\Projects_for_ alGam3aa\cd_dataset\training_set",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r"C:\Users\momag\OneDrive\Desktop\Projects_for_ alGam3aa\cd_dataset\testing_set",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#Build CNN Model

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

cnn.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Training the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',     
    patience=5,             
    restore_best_weights=True  
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,             
    patience=3,             
    min_lr=1e-6            
)

cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=50,
    callbacks=[early_stop, reduce_lr]
)

#Training the CNN Evaluate Model

loss, accuracy = cnn.evaluate(test_set)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# Making a single prediction
test_dir = r"C:\Users\momag\OneDrive\Desktop\Projects_for_ alGam3aa\cd_dataset\testing_set"

random_class = random.choice(os.listdir(test_dir))

class_path = os.path.join(test_dir, random_class)
random_image_file = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, random_image_file)

original_image = Image.open(image_path)

test_image = image.load_img(image_path, target_size=(64, 64))
img_array = image.img_to_array(test_image)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalization

result = cnn.predict(img_array)

predicted_label = 'Dog' if result[0][0] > 0.5 else 'Cat'

plt.imshow(original_image)
plt.title(f"Prediction: {predicted_label}")
plt.axis('off')
plt.show()

# Save And Load Model
cnn.save('CNN_model.keras')

from keras.models import load_model
cnn = load_model('CNN_model.keras')