

!pip install tensorflow[and-cuda] numpy

import tensorflow as tf

import os
import zipfile
import tempfile
import random
import numpy as np                                                                               # Importing numpy for Matrix Operations
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg                                                                              # Importing pandas to read CSV files
import matplotlib.pyplot as plt                                                                  # Importting matplotlib for Plotting and visualizing images
import math                                                                                      # Importing math module to perform mathematical operations
import cv2


# Tensorflow modules
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator                              # Importing the ImageDataGenerator for data augmentation
from tensorflow.keras.models import Sequential                                                   # Importing the sequential module to define a sequential model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization # Defining all the layers to build our CNN Model
from tensorflow.keras.optimizers import Adam,SGD                                                 # Importing the optimizers which can be used in our model
from sklearn import preprocessing                                                                # Importing the preprocessing module to preprocess the data
from sklearn.model_selection import train_test_split                                             # Importing train_test_split function to split the data into train and test
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16                                               # Importing confusion_matrix to plot the confusion matrix

# Display images using OpenCV
from google.colab.patches import cv2_imshow

#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report
from sklearn.metrics import mean_squared_error as mse                                                 # Importing cv2_imshow from google.patches to display images

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')



tf.keras.utils.set_random_seed(42)


import os
import zipfile
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def show_validation_predictions(model, val_gen, label_map, num_images=10):
    # Get mapping: index → folder name (e.g., 0 → '1')
    index_to_folder = {v: k for k, v in val_gen.class_indices.items()}

    # Fetch enough images from generator
    images = []
    labels = []
    for i in range((num_images // val_gen.batch_size) + 1):
        x_batch, y_batch = next(val_gen)
        images.extend(x_batch)
        labels.extend(y_batch)
        if len(images) >= num_images:
            break

    images = np.array(images[:num_images])
    labels = np.array(labels[:num_images])

    # Predict
    predictions = model.predict(images)
    predicted_indices = np.argmax(predictions, axis=1)
    actual_indices = np.argmax(labels, axis=1)

    # Plot
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        true_folder = index_to_folder[actual_indices[i]]
        pred_folder = index_to_folder[predicted_indices[i]]
        true_label = label_map.get(true_folder, "Unknown")
        pred_label = label_map.get(pred_folder, "Unknown")
        plt.title(f"True:\n{true_label}\nPred:\n{pred_label}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()




# === GLOBAL TEST DATA ===
val_gen = None

# === LABEL MAP (Folder name → Diagnosis) ===
label_map = {
    '0': 'Nasal polyps',
    '1': '⁠Deviated nasal septum',
    '2': 'Juvenile nasopharyngeal angiofibroma'
}

# === STEP 1: Extract ZIP ===
def extract_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

# === STEP 2: Image Generators ===
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(dataset_dir, image_size=(224, 224), batch_size=32):
    global val_gen

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen


# === STEP 3: Build Model ===
def create_vgg16_model(num_classes, input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === STEP 4: Plot Sample Images with Label Mapping ===
def plot_sample_images_from_zip(zip_path, num_images=5):
    extracted_path = extract_zip(zip_path)

    for item in os.listdir(extracted_path):
        potential_dataset_path = os.path.join(extracted_path, item)
        if os.path.isdir(potential_dataset_path):
            dataset_dir = potential_dataset_path
            break
    else:
        raise ValueError("No dataset folder found in ZIP.")

    image_paths = []
    class_labels = []
    for class_folder in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_folder)
        if os.path.isdir(class_dir):
            png_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.png')]
            for png_file in png_files[:num_images]:
                image_paths.append(os.path.join(class_dir, png_file))
                readable_label = label_map.get(class_folder, class_folder)
                class_labels.append(readable_label)

    plt.figure(figsize=(15, 5))
    for i, (img_path, label) in enumerate(zip(image_paths[:num_images], class_labels[:num_images])):
        imgread = mpimg.imread(img_path)
        imageFinal = cv2.resize(imgread, (224,224))
        plt.subplot(1, num_images, i + 1)
        plt.imshow(imageFinal)
        plt.title(label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# === STEP 5: Train From ZIP ===
def train_from_zip(zip_path, epochs=10):
    extracted_path = extract_zip(zip_path)

    for item in os.listdir(extracted_path):
        potential_dataset_path = os.path.join(extracted_path, item)
        if os.path.isdir(potential_dataset_path):
            dataset_dir = potential_dataset_path
            break
    else:
        raise ValueError("No dataset folder found in ZIP.")

    train_gen = create_generators(dataset_dir)
    model = create_vgg16_model(num_classes=train_gen.num_classes)

    # === Print class indices mapped to label names ===
    print("\nClass indices:")
    for folder_name, index in train_gen.class_indices.items():
        print(f"{index}: {label_map.get(folder_name, folder_name)}")

    # === Train the model ===
    model.fit(train_gen, epochs=epochs, validation_data=val_gen)
    return model

# === MAIN ===
if __name__ == "__main__":
    zip_file_path = '/content/drive/MyDrive/BHAI/NoseModel/NoseData.zip'  # Replace with your ZIP path

    print("📸 Plotting sample PNG images:")
   # plot_sample_images_from_zip(zip_file_path, num_images=5)

    print("\n🧠 Starting training with VGG16:")
    model = train_from_zip(zip_file_path, epochs=10)

    print("\n✅ Evaluating on validation data:")
    loss, acc = model.evaluate(val_gen)
    print(f"Validation Accuracy: {acc:.2f}")