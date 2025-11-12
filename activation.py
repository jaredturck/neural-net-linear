import math

class Activation:
    ''' Activation functions '''

    LAMBDA = 1.0507
    ALPHA = 1.67326

    @staticmethod
    def relu(x_array):
        ''' Applies Rectified Linear Unit (ReLU) - https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/ '''
        return [max(0, x) for x in x_array]

    @staticmethod
    def sigmoid(x_array):
        ''' Applies Sigmoid activation function - https://en.wikipedia.org/wiki/Sigmoid_function '''
        return [1 / (1 + (math.e ** -x)) for x in x_array]

    @staticmethod
    def selu(x_array):
        ''' Applies the Scaled Exponential Linear Unit (SELU) - https://www.geeksforgeeks.org/deep-learning/selu-activation-function-in-neural-network/ '''
        output = []
        for x in x_array:
            if x > 0:
                output.append(Activation.LAMBDA * x)
            else:
                output.append(Activation.LAMBDA * Activation.ALPHA * ((math.e ** x) - 1))
        return output

    @staticmethod
    def gelu(x_array):
        ''' Applies the Gaussian Error Linear Unit (GELU) activation function - https://arxiv.org/pdf/1606.08415 '''
        return [0.5 * x * (1 + math.erf(x / math.sqrt(2))) for x in x_array]

    @staticmethod
    def tanh(x_array):
        ''' Applies Hyperbolic Tangent (tanh) - https://www.geeksforgeeks.org/deep-learning/tanh-activation-in-neural-network/ '''
        return [math.tanh(x) for x in x_array]

    @staticmethod
    def softplus(x_array):
        ''' Applies the Softplus activation function - https://www.geeksforgeeks.org/deep-learning/softplus-function-in-neural-network/ '''
        return [math.log(1 + (math.e ** x)) for x in x_array]

    @staticmethod
    def softmax(x_array):
        ''' Applies Softmax - https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/ '''
        max_x = max(x_array)
        exp_values = [math.e ** (x - max_x) for x in x_array]
        total = sum(exp_values)
        return [exp / total for exp in exp_values]
