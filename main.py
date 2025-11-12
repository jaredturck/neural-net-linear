from layers import Layer
from activation import Activation
from loss import Loss
from backprop import Backpropogation

class Module:
    def __init__(self):

        self.layer_1 = Layer(10, 8, activation=Activation.relu)
        self.layer_2 = Layer(8, 8, activation=Activation.relu)
        self.layer_3 = Layer(8, 4, activation=Activation.softmax)

    def forward(self, x):
        ''' Forward pass '''

        x = self.layer_1.Linear(x)
        x = self.layer_2.Linear(x)
        x = self.layer_3.Linear(x)
        return x
    
    def backward(self, y):
        ''' Backwards pass '''
        
        l2_deltas = Backpropogation.compute_softmax_gradients(self.layer_3, y)
        l1_deltas = Backpropogation.compute_relu_gradients(self.layer_2, l2_deltas)
        in_deltas = Backpropogation.compute_relu_gradients(self.layer_1, l1_deltas)
    
    def train(self):
        ''' Train model '''

        x = [0.5, 0.2, 0.8, 0.1, 0.7, 0.3, 0.9, 0.4, 0.6, 0.2]
        y = [0, 0, 1, 0]

        for epoch in range(100):
            pred = self.forward(x)
            loss = Loss.categorical_cross_entropy(y, pred)

            self.backward(y)
            # self.optimizer_step()
