
class Optimizer:
    ''' Update weights and bias using optmizer '''

    @staticmethod
    def SGD(*layers, learning_rate=0.01):
        ''' Stochastic Gradient Descent optimizer '''

        for layer in layers:
            for k in range(layer.output_neurons):
                for j in range(layer.input_neurons):
                    layer.weights[k][j] -= learning_rate * layer.gradients[k][j]
                layer.bias[k] -= learning_rate * layer.bias_gradients[k]
