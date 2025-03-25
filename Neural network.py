# Import necessary libraries
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error (MSE) loss function
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Basic Neural Network class
class BasicNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.bias_output = np.random.randn(1, output_size)

    # Forward pass
    def forward(self, X):
        # Hidden layer calculation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Output layer calculation
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    # Backward pass and weights update
    def backward(self, X, y, output, learning_rate):
        # Calculate error
        output_error = y - output  # Error at the output
        output_delta = output_error * sigmoid_derivative(output)  # Delta for output layer

        # Error and delta for hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases using gradient descent
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output, learning_rate)
            
            # Calculate and display the loss at every 100th epoch
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")

# XOR dataset (used to test the neural network)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y = np.array([[0], [1], [1], [0]])  # Expected output (XOR)

# Create a neural network instance with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Train the neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the neural network with the XOR dataset
print("\nTesting the trained neural network:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {nn.forward(X[i])}, Actual Output: {y[i]}")
