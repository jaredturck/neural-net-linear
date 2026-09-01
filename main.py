import ctypes

F_RELU = 0
F_SIGMOID = 1
F_SELU = 2
F_GELU = 3
F_TANH = 4
F_SOFTPLUS = 5
F_SOFTMAX = 6

FLOAT = 0
INTEGER = 1
TEXT = 2
LABEL = 3

BYTE = 0

SEQUENTIAL = 0
SHUFFLED = 1
RANDOM = 2

lib = ctypes.CDLL('./nn/bin/main.so')

lib.create_model.argtypes = []
lib.create_model.restype = ctypes.c_void_p
lib.add_linear.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.add_linear.restype = ctypes.c_int
lib.train_loader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
lib.train_loader.restype = ctypes.c_int
lib.forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.forward.restype = None
lib.free_model.argtypes = [ctypes.c_void_p]
lib.free_model.restype = None

lib.create_dataset.argtypes = []
lib.create_dataset.restype = ctypes.c_void_p
lib.load_csv_dataset.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_char,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
lib.load_csv_dataset.restype = ctypes.c_int
lib.load_text_dataset.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.load_text_dataset.restype = ctypes.c_int
lib.dataset_size.argtypes = [ctypes.c_void_p]
lib.dataset_size.restype = ctypes.c_int
lib.dataset_input_size.argtypes = [ctypes.c_void_p]
lib.dataset_input_size.restype = ctypes.c_int
lib.dataset_output_size.argtypes = [ctypes.c_void_p]
lib.dataset_output_size.restype = ctypes.c_int
lib.dataset_token_count.argtypes = [ctypes.c_void_p]
lib.dataset_token_count.restype = ctypes.c_int
lib.dataset_vocab_size.argtypes = [ctypes.c_void_p]
lib.dataset_vocab_size.restype = ctypes.c_int
lib.dataset_encode_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
lib.dataset_encode_text.restype = None
lib.dataset_argmax.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.dataset_argmax.restype = ctypes.c_int
lib.dataset_label.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.dataset_label.restype = ctypes.c_char_p
lib.free_dataset.argtypes = [ctypes.c_void_p]
lib.free_dataset.restype = None

lib.create_row_sampler.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]
lib.create_row_sampler.restype = ctypes.c_void_p
lib.create_token_sampler.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
lib.create_token_sampler.restype = ctypes.c_void_p
lib.free_sampler.argtypes = [ctypes.c_void_p]
lib.free_sampler.restype = None

lib.create_dataloader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.create_dataloader.restype = ctypes.c_void_p
lib.dataloader_input_size.argtypes = [ctypes.c_void_p]
lib.dataloader_input_size.restype = ctypes.c_int
lib.dataloader_output_size.argtypes = [ctypes.c_void_p]
lib.dataloader_output_size.restype = ctypes.c_int
lib.free_dataloader.argtypes = [ctypes.c_void_p]
lib.free_dataloader.restype = None


def _columns(value):
    if isinstance(value, int):
        return [value]
    return list(value)


def _int_array(values):
    return (ctypes.c_int * len(values))(*values)


class Model:
    def __init__(self):
        self.model = lib.create_model()
        self.output_size = 0

    def add_linear(self, input_size, output_size, activation):
        if not lib.add_linear(self.model, input_size, output_size, activation):
            raise ValueError('Linear layer dimensions do not match the model')
        self.output_size = output_size

    def forward(self, x):
        output = (ctypes.c_float * self.output_size)()
        lib.forward(self.model, x, output)
        return output

    def free(self):
        lib.free_model(self.model)
        self.model = None


class Dataset:
    def __init__(self):
        self.dataset = lib.create_dataset()
        self.size = 0
        self.input_size = 0
        self.output_size = 0
        self.token_count = 0
        self.vocab_size = 0

    def _refresh(self):
        self.size = lib.dataset_size(self.dataset)
        self.input_size = lib.dataset_input_size(self.dataset)
        self.output_size = lib.dataset_output_size(self.dataset)
        self.token_count = lib.dataset_token_count(self.dataset)
        self.vocab_size = lib.dataset_vocab_size(self.dataset)

    def csv(self, path, x, y, types, widths=None, delimiter=',', header=False):
        x = _columns(x)
        y = _columns(y)
        widths = widths or {}

        if len(delimiter.encode()) != 1:
            raise ValueError('delimiter must be one byte')

        try:
            x_types = [types[column] for column in x]
            y_types = [types[column] for column in y]
        except KeyError as error:
            raise ValueError(f'missing type for column {error.args[0]}') from None

        x_widths = [widths.get(column, 0 if types[column] == TEXT else 1) for column in x]
        y_widths = [widths.get(column, 0 if types[column] == TEXT else 1) for column in y]

        success = lib.load_csv_dataset(
            self.dataset,
            path.encode(),
            delimiter.encode(),
            _int_array(x),
            _int_array(x_types),
            _int_array(x_widths),
            len(x),
            _int_array(y),
            _int_array(y_types),
            _int_array(y_widths),
            len(y),
            int(header),
        )
        if not success:
            raise RuntimeError(f'Failed to load CSV dataset: {path}')
        self._refresh()
        return self

    def text(self, path, tokenizer=BYTE):
        if not lib.load_text_dataset(self.dataset, path.encode(), tokenizer):
            raise RuntimeError(f'Failed to load text dataset: {path}')
        self._refresh()
        return self

    def encode(self, text):
        if self.input_size <= 0:
            raise ValueError('encode() requires a table dataset with a text X column')
        output = (ctypes.c_float * self.input_size)()
        lib.dataset_encode_text(self.dataset, text.encode(), output)
        return output

    def label(self, values):
        index = lib.dataset_argmax(self.dataset, values)
        label = lib.dataset_label(self.dataset, index)
        return label.decode() if label else str(index)

    def free(self):
        lib.free_dataset(self.dataset)
        self.dataset = None


class Sampler:
    def __init__(self, sampler):
        if not sampler:
            raise RuntimeError('Failed to create sampler')
        self.sampler = sampler

    def free(self):
        lib.free_sampler(self.sampler)
        self.sampler = None


class SequentialSampler(Sampler):
    def __init__(self, dataset, seed=1):
        super().__init__(lib.create_row_sampler(dataset.dataset, SEQUENTIAL, seed))


class RandomSampler(Sampler):
    def __init__(self, dataset, replacement=False, seed=1):
        strategy = RANDOM if replacement else SHUFFLED
        super().__init__(lib.create_row_sampler(dataset.dataset, strategy, seed))


class TokenSampler(Sampler):
    def __init__(self, dataset, sequence_length, strategy=SHUFFLED, samples=0, seed=1):
        super().__init__(lib.create_token_sampler(
            dataset.dataset,
            sequence_length,
            strategy,
            samples,
            seed,
        ))


class DataLoader:
    def __init__(self, dataset, sampler, batch_size=1, drop_last=False):
        self.loader = lib.create_dataloader(
            dataset.dataset,
            sampler.sampler,
            batch_size,
            int(drop_last),
        )
        if not self.loader:
            raise RuntimeError('Failed to create data loader')
        self.input_size = lib.dataloader_input_size(self.loader)
        self.output_size = lib.dataloader_output_size(self.loader)

    def free(self):
        lib.free_dataloader(self.loader)
        self.loader = None


def train(model, loader, epochs, learning_rate):
    if not lib.train_loader(model.model, loader.loader, epochs, learning_rate):
        raise RuntimeError('Training failed: model and loader shapes may not match')


class AnimalModel(Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_linear(input_size, 16, F_RELU)
        self.add_linear(16, output_size, F_SOFTMAX)


if __name__ == '__main__':
    dataset = Dataset().csv(
        'train.txt',
        x=[0],
        y=[1],
        types={0: TEXT, 1: LABEL},
    )

    sampler = RandomSampler(dataset, seed=42)
    loader = DataLoader(dataset, sampler, batch_size=16)
    model = AnimalModel(loader.input_size, loader.output_size)

    print('[+] Training started')
    train(model, loader, 5000, 0.001)

    for animal in ['cat', 'spider', 'salmon']:
        prediction = model.forward(dataset.encode(animal))
        print(f'{animal}: {dataset.label(prediction)}')

    model.free()
    loader.free()
    sampler.free()
    dataset.free()
