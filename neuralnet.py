import numpy as np
import random
import os
os.chdir('D:\\Python\\NeuralNetwork\\')

class neuralnet:

    def __init__(self, sizes):
        # Initialization of the neural network
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
    
    def training(self, traindata, batchsize, epochs, eta):
        n = len(traindata)
        # Train the network in specified num of epochs
        for i in range(epochs):
            random.shuffle(traindata)
            # Create batches with specified size
            mini_batches = [traindata[i:i+batchsize] for i in range(0, n, batchsize)]
            # Batchwise update weight and matrices of the network
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)

    def update(self, batch, eta):
        # Initialize lists of arrays for summing up gd_b and gd_w for full batch
        sum_gd_b = [np.zeros(b.shape) for b in self.biases]
        sum_gd_w = [np.zeros(w.shape) for w in self.weights]
        # Loop through all examples of the batch
        for x, y in batch:
            # Calculate gd_b and gd_w of the example
            gd_b, gd_w = self.backprop(x, y)
            # Add gd_b and gd_w to the arrays for the full batch
            sum_gd_b = [sum+gd for sum, gd in zip(sum_gd_b, gd_b)]
            sum_gd_w = [sum+gw for sum, gw in zip(sum_gd_w, gd_w)]
        # Update biases and weights of the network
        self.biases = [b-(eta/len(batch))*gdb 
                       for b, gdb in zip(self.biases, sum_gd_b)]
        self.weights = [w-(eta/len(batch))*gdw 
                        for w, gdw in zip(self.weights, sum_gd_w)]

    def backprop(self, input, target):
        # Initialize lists for Feed forward
        al = input
        a = [al]
        z = []
        # Feed forward
        for b, w in zip(self.biases, self.weights):
            zl = np.dot(w, al) + b
            z.append(zl)
            al = self.sigmoid(zl)
            a.append(al)
        # Initialize lists for Backpropagation
        gd_b = [np.zeros(b.shape) for b in self.biases]
        gd_w = [np.zeros(w.shape) for w in self.weights]
        # Calculate error at last layer
        gd_a = self.cost_derivative(a[-1], target)
        a_z = self.sigmoid_derivative(z[-1])
        delta = gd_a * a_z
        # Calculate gd_b and gd_w at last layer
        gd_b[-1] = delta
        gd_w[-1] = np.dot(delta, a[-2].transpose())
        # Backpropagation
        for L in range(2, self.num_layers):
            # Calculate the error at current layer
            gd_a = np.dot(self.weights[-L+1].transpose(), delta)
            a_z = self.sigmoid_derivative(z[-L])
            delta = gd_a * a_z
            # Calculate gd_b and gd_w at current layer
            gd_b[-L] = delta
            gd_w[-L] = np.dot(delta, a[-L-1].transpose())
        return (gd_b, gd_w)

    def predict(self, input):
        a = input
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.sigmoid(z)
        return a

    def evaluate(self, testdata):
        prediction_arrays = [self.predict(x[0]) for x in testdata]
        predictions = [x.argmax() for x in prediction_arrays]
        targets = [x[1] for x in testdata]

        n = len(testdata)
        n_c = 0
        for p, t in zip(predictions, targets):
            if p == t:  n_c += 1
            print(p, t, p==t)
        print('{0} of {1} test examples have been labeled correctly.'.format(n_c, n))
        print('The accuracy is {0:1%}.'.format(n_c/n))

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def cost_derivative(self, output, target):
        return output - target


# Load data
train = np.genfromtxt('mnist_train.csv', delimiter=',')
test = np.genfromtxt('mnist_test.csv', delimiter=',')

# Create list of trainings data tuples
def vectorize_target(z):
    tg_array = np.zeros((10, 1)) + 0.01
    tg_array[int(z)] = 0.99
    return tg_array
train_targets = [vectorize_target(t) for t in train[:,0]]
train_features = [f.reshape(784, 1) / 255 * 0.99 + 0.01 for f in train[:,1:]]
train_data = list(zip(train_features, train_targets))

# Create list of test data tuples
test_targets = [int(t) for t in test[:,0]]
test_features = [f.reshape(784, 1) for f in test[:,1:]]
test_data = list(zip(test_features, test_targets))

# Initialize and train the neural network
n = neuralnet([784, 30, 10])
n.training(train_data, 30, 10, 3.0)
n.evaluate(test_data)