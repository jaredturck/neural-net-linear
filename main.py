from datasets import Datasets
import ctypes, itertools

F_RELU = 0
F_SIGMOID = 1
F_SELU = 2
F_GELU = 3
F_TANH = 4
F_SOFTPLUS = 5
F_SOFTMAX = 6

class Model:
    def __init__(self, input_size):
        self.lib = ctypes.CDLL('./nn/bin/main.so')
        self.lib.create_model.argtypes = [ctypes.c_int]
        self.lib.create_model.restype = ctypes.c_void_p
        self.lib.add_linear.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.add_linear.restype = None
        self.lib.train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float
        ]
        self.lib.train.restype = None
        self.lib.forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.forward.restype = None
        self.lib.free_model.argtypes = [ctypes.c_void_p]
        self.lib.free_model.restype = None

        self.model = self.lib.create_model(input_size)
        self.input_size = input_size
        self.output_size = input_size

    def add_linear(self, output_size, activation):
        self.lib.add_linear(self.model, output_size, activation)
        self.output_size = output_size

    def train(self, train_x, train_y, epochs, learning_rate):
        flat_x = list(itertools.chain.from_iterable(train_x))
        flat_y = list(itertools.chain.from_iterable(train_y))
        x_c = (ctypes.c_float * len(flat_x))(*flat_x)
        y_c = (ctypes.c_float * len(flat_y))(*flat_y)

        self.lib.train(self.model, x_c, y_c, len(train_x), epochs, learning_rate)

    def forward(self, x):
        x_c = (ctypes.c_float * len(x))(*x)
        output_c = (ctypes.c_float * self.output_size)()
        self.lib.forward(self.model, x_c, output_c)
        return list(output_c)

    def free(self):
        self.lib.free_model(self.model)

if __name__ == '__main__':
    dataset = Datasets()
    dataset.animal_families_dataset()

    input_size = len(dataset.train_x[0])
    output_size = len(dataset.train_y[0])

    model = Model(input_size)
    model.add_linear(16, F_RELU)
    model.add_linear(output_size, F_SOFTMAX)

    print('[+] Training started')
    model.train(dataset.train_x, dataset.train_y, 5000, 0.001)

    for animal in ['cat', 'spider', 'salmon']:
        x = dataset.tokenize(animal)[:input_size]
        x = x + ([0] * (input_size - len(x)))
        pred = model.forward(x)
        pred_label = dataset.labels[dataset.argmax(pred)]
        print(f'{animal}: {pred_label}')

    model.free()
