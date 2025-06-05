import os
import cv2
# import random
import glob as gb
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix,f1_score

import tensorflow as tf
from tensorflow.keras import layers, models
# from keras.models import load_model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D

base_dir = 'dataset/' 
train_data = 'dataset/train'
test_data = 'dataset/test'

print(f'Folders of dataset: {os.listdir(base_dir)}')
print(f'Train data set: {os.listdir(train_data)}')
print(f'Test data set: {os.listdir(test_data)}')

for dir in os.listdir(train_data):
    if dir in ['Closed', 'Open']:
        full_path = os.path.join(train_data, dir)
        file = gb.glob(pathname = str(full_path + '/*.jpg'))
        print(f'lenght of files {len(file)} in {dir}')
    
for dir in os.listdir(test_data):
    if dir in ['Closed', 'Open']:
        full_path = os.path.join(test_data, dir)
        file = gb.glob(pathname = str(full_path + '/*.jpg'))
        print(f'lenght of files {len(file)} in {dir}')
    
def show_image(train_data):
    for dir in os.listdir(train_data):  
        full_path = os.path.join(train_data, dir)
        if os.path.isdir(full_path):  
            files = os.listdir(full_path)  
            for file in files:
                if file.endswith('.jpg'): 
                    img_path = os.path.join(full_path, file)
                    image = plt.imread(img_path)  
                    plt.imshow(image) 
                    plt.title(f"Directory: {dir}")  
                    plt.axis('off')  
                    plt.show()  
                    break  

code = {'Closed':0, 'Open':1}
def getCode(n):
    for x,y in code.items():
        if n==y:
            return x


size = []
for dir in os.listdir(train_data):
    if dir in ['Closed', 'Open']:
        full_path = os.path.join(train_data,dir)
        files = gb.glob(pathname = str(full_path + '/*.jpg'))
        for file in files:
            image = plt.imread(file)
            size.append(image.shape)
pd.Series(size).value_counts

s = 224
X_train = []
y_train = []
for dir in os.listdir(train_data):
    if dir in ['Closed', 'Open']:
        full_path = os.path.join(train_data,dir)
        files = gb.glob(pathname = str(full_path + '/*.jpg'))
        for file in files:
            print(f'Processing {file}')
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image, (s,s))
            X_train.append(image_array)
            y_train.append(code.get(dir))
        
print(f'lenght of train dataset: {len(X_train)}')
print(f'lenght of train label dataset: {len(y_train)}')

s=224
X_test = []
y_test = []
for dir in os.listdir(test_data):
    if dir in ['Closed', 'Open']:
        full_path = os.path.join(test_data,dir)
        files = gb.glob(pathname = str(full_path + '/*.jpg'))
        for file in files:
            image = cv2.imread(file)
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image, (s,s))
            X_test.append(image_array)
            y_test.append(code.get(dir))
        
print(f'lenght of test dataset: {len(X_test)}')
print(f'lenght of test label dataset: {len(y_test)}')

#converting list into numpy array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

normalization_layer = layers.Normalization()

normalization_layer.adapt(X_train)

X_train = normalization_layer(X_train)
X_test = normalization_layer(X_test)

print("Actual y_train shape:", y_train.shape)
print("Actual y_test shape:", y_test.shape)

# Convert integer labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Print the new shape
print("After one-hot encoding, y_train shape:", y_train.shape)  # Expected: (2467, 4)
print("After one-hot encoding, y_test shape:", y_test.shape)    # Expected: (433, 4)

print(f"unique values of y_train: {np.unique(y_train)}")
print(f"unique values of y_test: {np.unique(y_test)}")

# Define CNN model
Drowsiness_detection = models.Sequential([

    # Conv Layer 1
    layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(pool_size=2),

    # Conv Layer 2
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Conv Layer 3
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Conv Layer 4
    layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Output Layer (2 classes with softmax)
    layers.Dense(2, activation='softmax') 
])

# Compile the model
Drowsiness_detection.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
    loss='categorical_crossentropy',  #  Correct for one-hot encoding
    metrics=['accuracy']
)

# Print model summary
Drowsiness_detection.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1, restore_best_weights=True)

# Reduce learning rate if validation accuracy does not improve
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Save the best model weights
# Save the best model weights in the new `.keras` format
model_checkpoint = ModelCheckpoint(filepath='eyes_only_model.keras',  #  Change .h5 → .keras
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   save_weights_only=False,  # Change to False to save full model
                                   mode='max',
                                   verbose=1)

# Define augmentation only for training data
datagen = ImageDataGenerator(
    rotation_range=20,       # Rotate images by ±20 degrees
    width_shift_range=0.2,   # Shift width by 20%
    height_shift_range=0.2,  # Shift height by 20%
    shear_range=0.2,         # Shearing transformations
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Flip images horizontally
    fill_mode='nearest'      # Fill missing pixels
)

# Apply augmentation only to training data
train_generator = datagen.flow(X_train, y_train, batch_size=16, shuffle=True)

# Train the model
history = Drowsiness_detection.fit(
    train_generator,  #  Augmented training data
    epochs=30,        # Train for 30 epochs
    validation_data=(X_test, y_test),  #  No augmentation on test data
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

Drowsiness_detection.save("drowsiness_eye_cnn_model.h5")

os.listdir("./")

# Evaluate on test data
test_loss, test_acc = Drowsiness_detection.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

Indexcode = {0: 'Closed', 1: 'Open'}
def getLabel(one_hot_vector):
    class_index = int(np.argmax(one_hot_vector))  # Get the class index
    print(f"DEBUG: class_index = {class_index}")  #  Print class index for debugging
    
    print(f"DEBUG: Raw prediction vector = {one_hot_vector}")  # Print raw output

    if class_index in Indexcode:
        return Indexcode[class_index]
    else:
        return "Unknown"  # Handle unexpected cases
    
# Pick an image from X_test
index = 157  # Change this index to test different images
img = X_test[index]

# Display the image
plt.imshow(img)
plt.axis("off")  # Hide axes for clarity
plt.show()

# Reshape for prediction (add batch dimension)
img = np.expand_dims(img, axis=0)  # Shape becomes (1, 224, 224, 3)

# Make prediction
prediction = Drowsiness_detection.predict(img)

# Get predicted class label
predicted_label = getLabel(prediction)

# Print predicted class
print(f"Predicted Class: {predicted_label}")

# Convert to Active/Drowsy decision
if predicted_label in ["Closed"]:  # Closed or Yawn means Drowsy
    print("Final Output: Drowsy")
elif predicted_label in ["Open"]:  # Open or No Yawn means Active
    print("Final Output: Active")
else:
    print("Final Output: Unknown")  # Handle unexpected cases