# Hand Sign Recognition Using CNN

This project aims to build a Convolutional Neural Network (CNN) model to recognize hand signs based on the **Sign Language MNIST** dataset. The dataset contains grayscale images of hand signs representing letters from A-Z (26 classes). The model uses a CNN architecture to process the images and predict the correct hand sign (letter).

## Project Overview

The goal of this project is to classify hand signs into one of the 26 letters (A-Z) using a Convolutional Neural Network (CNN). The project is implemented in Python using libraries such as Keras, TensorFlow, OpenCV, and Scikit-learn.

### Key Features:
- **Dataset**: The dataset used in this project is **Sign Language MNIST** which consists of 28x28 pixel grayscale images of hand signs.
- **Model**: The model is built using a Convolutional Neural Network (CNN) with layers such as Conv2D, MaxPooling2D, Dense, and Dropout.
- **Performance**: The CNN model achieves an accuracy of **99.92%** on the test set.

## Setup and Installation

To get started with the project, clone the repository and install the required dependencies.

- ### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/hand-sign-recognition-cnn.git
cd hand-sign-recognition-cnn
```
## Dependencies

The following libraries are required to run the project:

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install all dependencies with:

```bash
pip install numpy pandas keras tensorflow opencv-python matplotlib scikit-learn
