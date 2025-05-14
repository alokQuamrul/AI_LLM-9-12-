import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# Reshape images to include a channel dimension (grayscale)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Convert labels to categorical one-hot encoding
num_classes = 10
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Define the model architecture
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
batch_size = 128
epochs = 5
model.fit(
    train_images, 
    train_labels, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_split=0.1
)

# Evaluate on test data
score = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# Save the model for later use
model.save("mnist_digit_predictor.h5")

# Function to predict a digit from an image
def predict_digit(image):
    """Predict a digit from a 28x28 grayscale image (values 0-1)"""
    # Reshape and normalize if needed
    if image.max() > 1:
        image = image / 255.0
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=(0, -1))
    elif len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    return np.argmax(prediction), prediction

# Test the predictor with a random test image
random_index = np.random.randint(0, len(test_images))
test_image = test_images[random_index]
test_label = np.argmax(test_labels[random_index])

predicted_digit, confidence = predict_digit(test_image)

# Display the test image and prediction
plt.imshow(test_image.squeeze(), cmap="gray")
plt.title(f"Actual: {test_label}, Predicted: {predicted_digit}")
plt.axis("off")
plt.show()

print(f"Predicted digit: {predicted_digit}")
print(f"Actual digit: {test_label}")
print("Confidence scores:", [f"{p:.2f}%" for p in confidence[0] * 100])