import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.error_history = []

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.connect(self.layers[-1])
        else:
            layer.connect()

        self.layers.append(layer)

    def compute(self, x):
        result = np.array(x)
        for layer in self.layers:
            result = layer.compute(result)
        return result

    def predict(self, x):
        result = []
        for xx in x:
            result.append(self.compute(xx))
        return np.array(result)

    def train(self, x, y, eta=0.01, threshold=1e-3, max_iters=None):

        x = np.array(x)
        y = np.array(y)
        train_set_size = len(x)
        index = 0
        count = 0
        error = np.array([100.0] * train_set_size)
        batch = 1
        while True:

            input = x[index]
            label = y[index]
            output = self.compute(input)
            d = label - output

            index = (index + 1) % len(x)
            count += 1
            error[index] = float(np.sqrt(np.dot(d, d)))
            mean_abs_error = np.mean(error)

            if count % train_set_size == 0:
                self.error_history.append(mean_abs_error)
                print("Training. Batch {:6d}. mean absolute error={:f}".format(batch, mean_abs_error))
                batch += 1

            if np.all(error < threshold) or (max_iters is not None and count > max_iters):
                break

            self.back_propagation(d)
            self.update(eta)

    def back_propagation(self, d):
        for layer in self.layers[::-1]:
            layer.back_propagation(d)

    def update(self, eta):
        for layer in self.layers:
            layer.update(eta)


class Layer:
    def __init__(self, number_of_neurons=10, input_size=5, activation="sigmoid"):
        self.number_of_neurons = number_of_neurons
        self.activation = activation
        self.neurons = []
        self.input_size = input_size
        self.next_layer = None

    def set_next_layer(self, layer):
        self.next_layer = layer

    def connect(self, last_layer=None):

        if last_layer is not None:
            self.input_size = last_layer.get_output_size()
            last_layer.set_next_layer(self)

        for i in np.arange(0, self.number_of_neurons):
            self.neurons.append(Neuron(self, i, self.activation))

    def get_output_size(self):
        return len(self.neurons)

    def compute(self, x):
        output = []
        for neuron in self.neurons:
            output.append(neuron.compute(x))
        return output

    def back_propagation(self, d):
        for neuron in self.neurons:
            neuron.back_propagation(d)

    def update(self, eta):
        for neuron in self.neurons:
            neuron.update(eta)


class Neuron:
    def __init__(self, layer, no, activation="sigmoid"):
        self.no = no
        self.layer = layer
        self.weights = np.array(np.random.rand(self.layer.input_size))
        self.activation = activation
        self.delta = 0.0
        self.activation_level = 0.0
        self.input = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.power(np.e, -x))

    @staticmethod
    def sigmoid_grad(x):
        return np.power(np.e, -x) / np.power(1 + np.power(np.e, -x), 2)

    def compute(self, x):
        self.input = x
        self.activation_level = np.dot(self.input, self.weights)
        if self.activation == "sigmoid":
            return Neuron.sigmoid(self.activation_level)
        else:
            return self.activation_level

    def back_propagation(self, d):
        if self.layer.next_layer is not None:
            tmp = 0.0
            for neuron in self.layer.next_layer.neurons:
                tmp += neuron.delta * neuron.weights[self.no]
        else:
            tmp = d[self.no]

        if self.activation == "sigmoid":
            self.delta = tmp * Neuron.sigmoid_grad(self.activation_level)
        else:
            self.delta = tmp

    def update(self, eta):
        self.weights += eta * self.delta * np.array(self.input)
