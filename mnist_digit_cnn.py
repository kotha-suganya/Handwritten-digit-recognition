import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


def main():

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values (0–255 → 0–1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape data for CNN (samples, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 digits (0–9)
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy:", test_accuracy)



    # Prediction example 
    index = 10
    prediction = model.predict(x_test[index].reshape(1,28,28,1))
    predicted_label = np.argmax(prediction)

    plt.imshow(x_test[index].reshape(28,28), cmap='gray')
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

#             # execution commands
# python -m venv venv 
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\venv\Scripts\activate
# pip install tensorflow matplotlib numpy
# python mnist_digit_cnn.py