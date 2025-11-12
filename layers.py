import random

class Layer:
    ''' Neural network layers '''

    def __init__(self, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.weights = []
        for _ in range(output_neurons):
            self.weights.append([random.random() for i in range(input_neurons)])

        self.bias = [random.random() for i in range(output_neurons)]
        self.backprop_cache = []
        self.gradients = []
        self.activation_function = None

    def Linear(self, x_array):
        ''' Linear layer '''
        
        output = []
        for i in range(self.output_neurons):
            y = sum(x * w for x,w in zip(x_array, self.weights[i])) + self.bias[i]
            output.append(y)
        
        logits = self.activation_function(output)
        self.backprop_cache = [x_array, output, logits]

        return logits
