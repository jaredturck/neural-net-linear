import math

class Loss:
    ''' Loss functions '''

    @staticmethod
    def mse(y_true, y_pred):
        ''' Mean Squared Error (MSE) - https://www.geeksforgeeks.org/maths/mean-squared-error/ '''
        output = []
        for y1, y2 in zip(y_true, y_pred):
            output.append((y1 - y2) ** 2)
        return sum(output) / len(output)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        ''' Binary Cross Entropy (BCE) - https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/ '''
        total = 0
        for y, p, in zip(y_true, y_pred):
            total += y * math.log(p) + (1 - y) * math.log(1 - p)
        return -total / len(y_true)
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        ''' Categorical Cross Entropy (CCE) - https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/ '''
        total = 0
        for y, p in zip(y_true, y_pred):
            total += y * math.log(p)
        return -total
