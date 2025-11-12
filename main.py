from layers import Layer
from activation import Activation
from loss import Loss
from backprop import Backpropogation
from optimizers import Optimizer
from datasets import Datasets
import math

class Module:
    def __init__(self):

        self.layer_1 = Layer(18, 16, activation=Activation.relu)
        self.layer_2 = Layer(16, 16, activation=Activation.relu)
        self.layer_3 = Layer(16, 10, activation=Activation.softmax)

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

        dataset = Datasets()
        dataset.animal_families_dataset()

        for epoch in range(1000):
            for x,y in dataset.train:
                pred = self.forward(x)
                scaler_loss = Loss.categorical_cross_entropy(y, pred)

                self.backward(y)
                Optimizer.SGD(self.layer_1, self.layer_2, self.layer_3, learning_rate=0.01)

                if epoch % 10 == 0:
                    print(f'Epoch {epoch+1}, loss: {scaler_loss}')

                if scaler_loss <= 0.01:
                    print('Training complete')
                    break

if __name__ == '__main__':
    model = Module()
    model.train()
