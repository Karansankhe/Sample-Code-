# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Set random seed for reproducibility
np.random.seed(42)

# Function to preprocess the drone orthophotos (resizing, normalization, etc.)
def preprocess_image(image, target_size=(256, 256)):
    """Preprocess input image: resize and normalize pixel values."""
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0  # Normalize pixel values to [0, 1]
    return image_normalized

# Load dataset (e.g., SVAMITVA dataset with labeled orthophotos)
def load_data(image_paths, labels):
    """Load and preprocess data from the SVAMITVA dataset."""
    X, y = [], []
    for img_path, label in zip(image_paths, labels):
        # Read image
        image = cv2.imread(img_path)
        # Preprocess image
        image = preprocess_image(image)
        # Append to list
        X.append(image)
        y.append(label)  # Label can be segmentation mask for buildings, roads, waterbodies
    return np.array(X), np.array(y)

# Create CNN-based model for image segmentation
def create_model(input_shape=(256, 256, 3), num_classes=3):
    """Create a CNN model for feature extraction and segmentation."""
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully connected layers for classification
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Output layer (num_classes represents building, road, waterbody classes)
    model.add(layers.Conv2D(num_classes, (1, 1), activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data augmentation (optional)
def augment_data(X_train, y_train):
    """Augment training data with rotations, shifts, flips, etc."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    return datagen

# Sample code to split dataset and train the model
def train_model(X, y, num_classes=3, batch_size=32, epochs=20):
    """Train the AI model on the preprocessed orthophoto dataset."""
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    
    # Create model
    model = create_model(num_classes=num_classes)
    
    # Optionally augment training data
    datagen = augment_data(X_train, y_train)
    
    # Train the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs)
    
    # Save the trained model
    model.save('feature_extraction_model.h5')
    
    return model, history

# Function to predict features from new orthophotos
def predict_features(model, image):
    """Use the trained model to predict features on a new orthophoto."""
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)
    return prediction

# Example usage
if __name__ == "__main__":
    # Assuming 'image_paths' contains file paths to drone images, and 'labels' contains ground truth masks
    image_paths = ['drone_image_1.png', 'drone_image_2.png']  # Example paths
    labels = [0, 1]  # Example labels (0: background, 1: building, etc.)
    
    # Load data
    X, y = load_data(image_paths, labels)
    
    # Train the model
    model, history = train_model(X, y)
    
    # Predict features from a new image
    test_image = cv2.imread('new_drone_image.png')
    prediction = predict_features(model, test_image)
    
    print("Prediction shape:", prediction.shape)  # Output will have the shape of the input image with predicted classes
