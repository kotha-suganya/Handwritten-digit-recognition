
MNIST Digit Classifier (CNN)
This project implements a Deep Learning model to identify handwritten digits (0-9) using the classic MNIST dataset. By using a Convolutional Neural Network (CNN) architecture, the model is able to extract spatial features from images, significantly outperforming standard dense neural networks.

Project Architecture
The model uses a sequential architecture with the following layers:

Convolutional Layers: Two Conv2D layers using 3x3 kernels and ReLU activation to detect edges and shapes.

Pooling: MaxPooling2D layers to reduce spatial dimensions and prevent overfitting.

Dense Head: A flattened layer followed by a 128-neuron ReLU layer and a final Softmax output for 10-class classification.

Performance
Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Training: 5 Epochs

Typical Accuracy: ~98.5% - 99.1% on the test set.

How to Run
1. Setup Environment
python -m venv venv
# Windows:
.\venv\Scripts\activate


Requirements:

Python 3.8+

TensorFlow 2.x

Matplotlib

NumPy
