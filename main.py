from layers import Layer
from activation import Activation
from loss import Loss

class Module:
    def __init__(self):

        self.layer_1 = Layer(10, 8)
        self.layer_2 = Layer(8, 8)
        self.layer_3 = Layer(8, 4)

    def forward(self, x):
        ''' Forward pass '''

        x = Activation.relu(self.layer_1.Linear(x))
        x = Activation.relu(self.layer_2.Linear(x))
        x = Activation.softmax(self.layer_3.Linear(x))
        return x
    
    def train(self):
        ''' Train model '''

        x = [0.5, 0.2, 0.8, 0.1, 0.7, 0.3, 0.9, 0.4, 0.6, 0.2]
        y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        for epoch in range(100):
            pred = self.forward(x)
            loss = Loss.categorical_cross_entropy(y, pred)

            # self.backward()
            # self.optimizer_step()
