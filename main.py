from datasets import Datasets
import ctypes, itertools

class Module:
    def __init__(self):
        self.lib = ctypes.CDLL('./nn/bin/main.so')
        self.lib.py_train.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.py_train.restype = None
        self.lib.py_predict.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.py_predict.restype = None
    
    def train(self):
        dataset = Datasets()
        dataset.animal_families_dataset()

        dataset_size = len(dataset.train_x)
        input_size = len(dataset.train_x[0])
        output_size = len(dataset.train_y[0])

        flat_x = list(itertools.chain.from_iterable(dataset.train_x))
        flat_y = list(itertools.chain.from_iterable(dataset.train_y))
        x_c = (ctypes.c_float * len(flat_x))(*flat_x)
        y_c = (ctypes.c_float * len(flat_y))(*flat_y)

        print('[+] Training started')
        self.lib.py_train(x_c, y_c, dataset_size, input_size, output_size)

    def predict(self):
        ''' Predict '''

        dataset = Datasets()
        dataset.animal_families_dataset()

        input_size = 18
        output_size = len(dataset.train_y[0])

        while True:
            animal = input('Enter animal: ').lower()

            x = dataset.tokenize(animal)[:18]
            x = x + ([0] * (input_size - len(x)))

            x_in = (ctypes.c_float * input_size)(*x)
            logits_c = (ctypes.c_float * output_size)()

            self.lib.py_predict(x_in, logits_c)
            pred = list(logits_c)

            pred_label = dataset.labels[dataset.argmax(pred)]
            print(pred_label)

if __name__ == '__main__':
    model = Module()
    model.train()
    model.predict()
