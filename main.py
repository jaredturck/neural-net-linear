import ctypes

F_RELU = 0
F_SIGMOID = 1
F_SELU = 2
F_GELU = 3
F_TANH = 4
F_SOFTPLUS = 5
F_SOFTMAX = 6

lib = ctypes.CDLL('./nn/bin/main.so')

lib.create_model.argtypes = [ctypes.c_int]
lib.create_model.restype = ctypes.c_void_p
lib.add_linear.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.add_linear.restype = None
lib.train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float
]
lib.train.restype = None
lib.forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.forward.restype = None
lib.free_model.argtypes = [ctypes.c_void_p]
lib.free_model.restype = None

lib.load_animal_dataset.argtypes = [ctypes.c_char_p]
lib.load_animal_dataset.restype = ctypes.c_void_p
lib.dataset_size.argtypes = [ctypes.c_void_p]
lib.dataset_size.restype = ctypes.c_int
lib.dataset_input_size.argtypes = [ctypes.c_void_p]
lib.dataset_input_size.restype = ctypes.c_int
lib.dataset_output_size.argtypes = [ctypes.c_void_p]
lib.dataset_output_size.restype = ctypes.c_int
lib.dataset_train_x.argtypes = [ctypes.c_void_p]
lib.dataset_train_x.restype = ctypes.POINTER(ctypes.c_float)
lib.dataset_train_y.argtypes = [ctypes.c_void_p]
lib.dataset_train_y.restype = ctypes.POINTER(ctypes.c_float)
lib.dataset_tokenize.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.dataset_tokenize.restype = None
lib.dataset_argmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.dataset_argmax.restype = ctypes.c_int
lib.dataset_label.argtypes = [ctypes.c_int]
lib.dataset_label.restype = ctypes.c_char_p
lib.free_dataset.argtypes = [ctypes.c_void_p]
lib.free_dataset.restype = None

class Model:
    def __init__(self, input_size):
        self.model = lib.create_model(input_size)
        self.output_size = input_size

    def add_linear(self, output_size, activation):
        lib.add_linear(self.model, output_size, activation)
        self.output_size = output_size

    def train(self, train_x, train_y, dataset_size, epochs, learning_rate):
        lib.train(self.model, train_x, train_y, dataset_size, epochs, learning_rate)

    def forward(self, x):
        output = (ctypes.c_float * self.output_size)()
        lib.forward(self.model, x, output)
        return output

    def free(self):
        lib.free_model(self.model)

class Dataset:
    def __init__(self, path):
        self.dataset = lib.load_animal_dataset(path.encode())
        self.size = lib.dataset_size(self.dataset)
        self.input_size = lib.dataset_input_size(self.dataset)
        self.output_size = lib.dataset_output_size(self.dataset)
        self.train_x = lib.dataset_train_x(self.dataset)
        self.train_y = lib.dataset_train_y(self.dataset)

    def tokenize(self, text):
        output = (ctypes.c_float * self.input_size)()
        lib.dataset_tokenize(text.encode(), output, self.input_size)
        return output

    def label(self, values):
        index = lib.dataset_argmax(values, self.output_size)
        return lib.dataset_label(index).decode()

    def free(self):
        lib.free_dataset(self.dataset)

if __name__ == '__main__':
    dataset = Dataset('train.txt')

    model = Model(dataset.input_size)
    model.add_linear(16, F_RELU)
    model.add_linear(dataset.output_size, F_SOFTMAX)

    print('[+] Training started')
    model.train(dataset.train_x, dataset.train_y, dataset.size, 5000, 0.001)

    for animal in ['cat', 'spider', 'salmon']:
        pred = model.forward(dataset.tokenize(animal))
        print(f'{animal}: {dataset.label(pred)}')

    model.free()
    dataset.free()
