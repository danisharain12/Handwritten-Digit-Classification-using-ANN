# Handwritten Digit Classification using ANN

This project implements an Artificial Neural Network (ANN) using TensorFlow/Keras to classify handwritten digits from the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It’s a great beginner-friendly project to understand deep learning fundamentals such as dense layers, activation functions, regularization, and model evaluation.

---

## Project Objective

To train a neural network that can accurately classify 28x28 grayscale images of digits (0–9) with minimal overfitting and efficient performance.

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

## Dataset

- **MNIST**: 60,000 training images and 10,000 testing images of handwritten digits.
- Loaded using:  
  ```python
  from tensorflow.keras.datasets import mnist

---

## Model Architecture

- model = Sequential()

# Flattening the 2D input images
- model.add(Flatten(input_shape=(28, 28)))

# First hidden layer with L2 regularization
- model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))

# Second hidden layer with dropout
- model.add(Dense(64, activation='relu'))
- model.add(Dropout(0.3))

# Third hidden layer with dropout
- model.add(Dense(32, activation='relu'))
- model.add(Dropout(0.2))

# Output layer with softmax for multi-class classification
- model.add(Dense(10, activation='softmax'))

---

## Why These Choices?

- Flatten Layer: Transforms 28x28 matrix into a 784-element vector.
- Dense Layers: Enable the model to learn complex patterns.
- ReLU Activation: Speeds up training and handles non-linearity.
- Dropout Layers: Reduce overfitting by randomly disabling neurons during training.
- L2 Regularization: Penalizes large weights, encouraging generalization.
- Softmax Output: Converts output to class probabilities.

---

## Results

- Validation Accuracy: ~97%
- Final Test Accuracy: 0.9728
- Plotted training & validation accuracy and loss over epochs.
- ![image](https://github.com/user-attachments/assets/a1880e40-fd5c-4945-ac21-945ee2c623d8)
- ![image](https://github.com/user-attachments/assets/26d34b71-15e6-4737-8825-c38796dc9d1d)

## Key Takeaways

- Simple ANN models can achieve high accuracy on structured image datasets like MNIST.
- Regularization (Dropout + L2) effectively combats overfitting.
- EarlyStopping is a must-have for training stability and efficiency.

## Contact
- Danish Karim
- danisharain253@gmail.com

