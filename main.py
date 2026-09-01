import ctypes

F_RELU = 0
F_SIGMOID = 1
F_SELU = 2
F_GELU = 3
F_TANH = 4
F_SOFTPLUS = 5
F_SOFTMAX = 6

lib = ctypes.CDLL('./nn/bin/main.so')

lib.create_model.argtypes = []
lib.create_model.restype = ctypes.c_void_p
lib.add_linear.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.add_linear.restype = ctypes.c_int
lib.train_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
lib.train_dataset.restype = None
lib.forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.forward.restype = None
lib.free_model.argtypes = [ctypes.c_void_p]
lib.free_model.restype = None

lib.create_dataset.argtypes = []
lib.create_dataset.restype = ctypes.c_void_p
lib.load_animal_dataset.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.load_animal_dataset.restype = ctypes.c_int
lib.dataset_size.argtypes = [ctypes.c_void_p]
lib.dataset_size.restype = ctypes.c_int
lib.dataset_input_size.argtypes = [ctypes.c_void_p]
lib.dataset_input_size.restype = ctypes.c_int
lib.dataset_output_size.argtypes = [ctypes.c_void_p]
lib.dataset_output_size.restype = ctypes.c_int
lib.dataset_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
lib.dataset_tokenize.restype = None
lib.dataset_argmax.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.dataset_argmax.restype = ctypes.c_int
lib.dataset_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.dataset_label.restype = ctypes.c_char_p
lib.free_dataset.argtypes = [ctypes.c_void_p]
lib.free_dataset.restype = None


class Model:
    def __init__(self):
        self.model = lib.create_model()
        self.output_size = 0

    def add_linear(self, input_size, output_size, activation):
        success = lib.add_linear(self.model, input_size, output_size, activation)
        if not success:
            raise ValueError('Linear layer dimensions do not match the model')
        self.output_size = output_size

    def forward(self, x):
        output = (ctypes.c_float * self.output_size)()
        lib.forward(self.model, x, output)
        return output

    def free(self):
        lib.free_model(self.model)


class Dataset:
    def __init__(self):
        self.dataset = lib.create_dataset()
        self.size = 0
        self.input_size = 0
        self.output_size = 0

    def load_animal(self, path):
        success = lib.load_animal_dataset(self.dataset, path.encode())
        if not success:
            raise RuntimeError(f'Failed to load dataset: {path}')

        self.size = lib.dataset_size(self.dataset)
        self.input_size = lib.dataset_input_size(self.dataset)
        self.output_size = lib.dataset_output_size(self.dataset)

    def tokenize(self, text):
        output = (ctypes.c_float * self.input_size)()
        lib.dataset_tokenize(self.dataset, text.encode(), output)
        return output

    def label(self, values):
        index = lib.dataset_argmax(self.dataset, values)
        return lib.dataset_label(self.dataset, index).decode()

    def free(self):
        lib.free_dataset(self.dataset)


def train(model, dataset, epochs, learning_rate):
    lib.train_dataset(model.model, dataset.dataset, epochs, learning_rate)


class AnimalModel(Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_linear(input_size, 16, F_RELU)
        self.add_linear(16, output_size, F_SOFTMAX)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_animal('train.txt')

    model = AnimalModel(dataset.input_size, dataset.output_size)

    print('[+] Training started')
    train(model, dataset, 5000, 0.001)

    for animal in ['cat', 'spider', 'salmon']:
        prediction = model.forward(dataset.tokenize(animal))
        print(f'{animal}: {dataset.label(prediction)}')

    model.free()
    dataset.free()
