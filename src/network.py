import random
import numpy as np

class Network:
    def __init__(self, sizes):
        """
        Initialize the neural network with the given sizes.
        
        Parameters:
        sizes (list): A list containing the number of neurons in each layer.
                      For example, [784, 30, 10] would create a network with
                      784 input neurons, one hidden layer with 30 neurons,
                      and an output layer with 10 neurons.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Biases and weights are initialized with Gaussian distribution with mean 0 and sd of 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the network if 'a' is input.
        
        Parameters:
        a (numpy array): The input to the network.
        
        Returns:
        numpy array: The output of the network.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        
        Parameters:
        training_data (list): A list of tuples '(x, y)' representing the training inputs
                              and the desired outputs.
        epochs (int): The number of epochs to train for.
        mini_batch_size (int): The size of the mini-batches to use when sampling.
        eta (float): The learning rate.
        test_data (list, optional): If provided, the network will be evaluated
                                    against the test data after each epoch, and
                                    partial progress will be printed out.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini-batch.
        
        Parameters:
        mini_batch (list): A list of tuples '(x, y)'.
        eta (float): The learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple representing the gradient for the cost function.
        
        Parameters:
        x (numpy array): The input to the network.
        y (numpy array): The desired output.
        
        Returns:
        tuple: (nabla_b, nabla_w), representing the gradients for the biases and weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x]  # List to store all the activations, layer by layer
        zs = []  # List to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Update the gradients for the previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs
        the correct result.
        
        Parameters:
        test_data (list): A list of tuples '(x, y)' where 'x' is the input and 'y' is the desired output.
        
        Returns:
        int: The number of test inputs for which the network is correct.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives ∂C/∂a for the output activations.
        
        Parameters:
        output_activations (numpy array): The output of the network.
        y (numpy array): The desired output.
        
        Returns:
        numpy array: The derivative of the cost function.
        """
        return (output_activations - y)
