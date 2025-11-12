from layers import Layer
from activation import Activation
from loss import Loss
from backprop import Backpropogation
from optimizers import Optimizer
from datasets import Datasets
import random, time

class Module:
    def __init__(self):

        self.layer_1 = Layer(18, 32, activation=Activation.relu)
        self.layer_2 = Layer(32, 32, activation=Activation.relu)
        self.layer_3 = Layer(32, 12, activation=Activation.softmax)

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
        start = time.time()

        for epoch in range(1000):
            avg_loss = 0
            random.shuffle(dataset.train)
            for x,y in dataset.train:
                pred = self.forward(x)
                avg_loss += Loss.categorical_cross_entropy(y, pred)

                self.backward(y)
                Optimizer.SGD(self.layer_1, self.layer_2, self.layer_3, learning_rate=0.001)

            avg_loss = avg_loss / len(dataset.train)
            if time.time() - start > 5:
                start = time.time()
                print(f'Epoch {epoch+1}, loss: {avg_loss}')

            if avg_loss <= 0.05:
                print('Training complete')
                return
    
    def predict(self):
        ''' Predict '''

        dataset = Datasets()

        while True:
            animal = input('Enter animal: ').lower()

            x = dataset.tokenize(animal)[:18]
            x = x + ([0] * (18 - len(x)))

            pred = self.forward(x)
            pred_label = dataset.labels[dataset.argmax(pred)]
            print(pred_label)

if __name__ == '__main__':
    model = Module()
    model.train()
    model.predict()
