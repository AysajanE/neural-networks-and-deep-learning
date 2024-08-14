"""
mnist_loader.py
~~~~~~~~~~~~~~~

A library to load the MNIST image data for handwritten digit recognition.
This module provides functions to load and preprocess the MNIST dataset,
making it ready for use in neural network training and testing.

The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
Each image is a 28x28 pixel grayscale image of a handwritten digit (0-9).

Key Functions:
- load_data(): Loads raw MNIST data
- load_data_wrapper(): Prepares data for neural network use
- vectorized_result(): Converts a digit to a vector representation

Dependencies:
- numpy: For efficient numerical computations
- gzip: For reading compressed data files
"""

import gzip
import pickle
import numpy as np

def load_data():
    """
    Load the MNIST dataset from a gzipped pickle file.
    
    Returns:
    tuple: Containing:
        - training_data: 50,000 images and labels for training
        - validation_data: 10,000 images and labels for validation
        - test_data: 10,000 images and labels for testing
    
    Each dataset is a tuple (X, y) where:
        - X is a numpy array with shape (n_samples, 784), 784 = 28*28 pixels
        - y is a numpy array with shape (n_samples,) containing labels (0-9)
    """
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        return pickle.load(f, encoding='latin1')

def load_data_wrapper():
    """
    Prepare MNIST data for neural network training and testing.
    
    Returns:
    tuple: Containing:
        - training_data: List of 50,000 tuples (x, y) where:
            x is a 784-dimensional numpy array (input)
            y is a 10-dimensional numpy array (desired output)
        - validation_data: List of 10,000 tuples (x, y) where:
            x is a 784-dimensional numpy array (input)
            y is the corresponding digit (0-9)
        - test_data: Same format as validation_data
    """
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # The wrapper keeps the labels as integers for validation and test data.
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Convert a digit (0-9) into a 10-dimensional unit vector.
    
    Args:
    j (int): The digit to be converted (0-9)
    
    Returns:
    numpy.ndarray: A 10-dimensional unit vector with a 1.0 in the j-th position
                   and zeroes elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


"""
mnist_loader.py Explanation
~~~~~~~~~~~~~~~
1. Module Overview:
   The `mnist_loader.py` module is designed to load and preprocess the MNIST (Modified National Institute of Standards and Technology) dataset. This dataset is a large collection of handwritten digits that is commonly used for training various image processing systems and machine learning algorithms.

2. MNIST Dataset Structure:
   - The dataset consists of 70,000 images of handwritten digits (0-9).
   - Each image is a 28x28 pixel grayscale image.
   - The dataset is divided into:
     - 50,000 training images
     - 10,000 validation images
     - 10,000 test images

3. Function: `load_data()`
   - Purpose: This function loads the raw MNIST data from a gzipped pickle file.
   - Process:
     a. Opens the file '../data/mnist.pkl.gz' using gzip (for decompression)
     b. Uses pickle to load the data (pickle is a Python module for serializing and deserializing Python objects)
   - Output: Returns a tuple containing three elements:
     1. Training data: 50,000 images and their labels
     2. Validation data: 10,000 images and their labels
     3. Test data: 10,000 images and their labels
   - Data format: Each dataset is a tuple (X, y) where:
     - X is a numpy array with shape (n_samples, 784), representing flattened 28x28 images
     - y is a numpy array with shape (n_samples,) containing the corresponding labels (0-9)

4. Function: `load_data_wrapper()`
   - Purpose: Prepares the MNIST data for use in neural network training and testing.
   - Process:
     a. Calls `load_data()` to get the raw data
     b. Processes each dataset (training, validation, test) as follows:
        - For inputs (images):
          - Reshapes each image from a 784-element array to a (784, 1) column vector
        - For training data outputs:
          - Converts each label to a 10-dimensional unit vector (using `vectorized_result()`)
        - For validation and test data outputs:
          - Keeps the original integer labels
     c. Combines inputs and outputs into tuples
   - Output: Returns a tuple containing:
     1. Training data: List of 50,000 tuples (x, y) where:
        - x is a 784-dimensional numpy array (input image)
        - y is a 10-dimensional numpy array (desired output)
     2. Validation data: List of 10,000 tuples (x, y) where:
        - x is a 784-dimensional numpy array (input image)
        - y is the corresponding digit (0-9)
     3. Test data: Same format as validation data

5. Function: `vectorized_result(j)`
   - Purpose: Converts a digit (0-9) into a 10-dimensional unit vector.
   - Process: Creates a zero vector of length 10 and sets the j-th element to 1.0
   - This is used for creating "one-hot" encodings of the digits, which is a common way to represent categorical data in neural networks.

6. Data Representation:
   - Each image is represented as a 784-dimensional vector (28 * 28 = 784 pixels)
   - Each pixel value is a grayscale intensity between 0 (white) and 1 (black)
   - Labels are represented as integers from 0 to 9 for validation and test data
   - For training data, labels are converted to 10-dimensional vectors (one-hot encoding)

This loader prepares the MNIST data in a format that's suitable for training a neural network to recognize handwritten digits. The images (inputs) are prepared as column vectors, and the labels (desired outputs) are prepared in a format that allows the network to learn the mapping between the input images and their corresponding digits.

In the context of recognizing handwritten digits:
- The input to our neural network will be a 784-dimensional vector representing a single digit image.
- The output of our network (for training data) will be a 10-dimensional vector, where the index of the highest value corresponds to the predicted digit.
"""