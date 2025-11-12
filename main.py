from layers import Layer
from activation import Activation
from loss import Loss

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

        # Layer 3 (softmax)
        x, z, a = self.layer_3.backprop_cache
        deltas = [ai - yi for ai, yi in zip(a,y)]

        self.layer_3.gradients = []
        for i in range(self.layer_3.output_neurons):
            self.layer_3.gradients.append([deltas[i] * xi for xi in x])
        
        l2_deltas = []
        for j in range(self.layer_3.input_neurons):
            l2_deltas.append(sum(self.layer_3.weights[i][j] * deltas[i] for i in range(self.layer_3.output_neurons)))
        
        # Layer 2 (relu)
        x, z, a = self.layer_2.backprop_cache
        mask = [int(v > 0) for v in z]
        deltas = [d * m for d,m in zip(l2_deltas, mask)]

        self.layer_2.gradients = []
        for i in range(self.layer_2.output_neurons):
            self.layer_2.gradients.append([deltas[i] * xi for xi in x])
        
        l1_deltas = []
        for j in range(self.layer_2.input_neurons):
            l1_deltas.append(sum(self.layer_2.weights[i][j] * deltas[i] for i in range(self.layer_2.output_neurons)))
        
        # Layer 1 (relu)
        x, z, a = self.layer_1.backprop_cache
        mask = [int(v > 0) for v in z]
        deltas = [d * m for d,m in zip(l1_deltas, mask)]

        self.layer_1.gradients = []
        for i in range(self.layer_1.output_neurons):
            self.layer_1.gradients.append([deltas[i] * xi for xi in x])
    
    def train(self):
        ''' Train model '''

        x = [0.5, 0.2, 0.8, 0.1, 0.7, 0.3, 0.9, 0.4, 0.6, 0.2]
        y = [0, 0, 1, 0]

        for epoch in range(100):
            pred = self.forward(x)
            loss = Loss.categorical_cross_entropy(y, pred)

            self.backward(y)
            # self.optimizer_step()
