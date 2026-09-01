import ctypes
from pathlib import Path

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

__all__ = [
    'F_RELU',
    'F_SIGMOID',
    'F_SELU',
    'F_GELU',
    'F_TANH',
    'F_SOFTPLUS',
    'F_SOFTMAX',
    'FLOAT',
    'INTEGER',
    'TEXT',
    'LABEL',
    'BYTE',
    'SEQUENTIAL',
    'SHUFFLED',
    'RANDOM',
    'Model',
    'Dataset',
    'Sampler',
    'SequentialSampler',
    'RandomSampler',
    'TokenSampler',
    'DataLoader',
    'train',
]


def _load_library():
    library_path = Path(__file__).resolve().parent / 'nn' / 'bin' / 'main.so'
    return ctypes.CDLL(str(library_path))


def _configure_library(lib):
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


_lib = _load_library()
_configure_library(_lib)


def _columns(value):
    if isinstance(value, int):
        return [value]
    return list(value)


def _int_array(values):
    return (ctypes.c_int * len(values))(*values)


class Model:
    def __init__(self):
        self._model = _lib.create_model()
        self.output_size = 0

    def add_linear(self, input_size, output_size, activation):
        if not _lib.add_linear(self._model, input_size, output_size, activation):
            raise ValueError('Linear layer dimensions do not match the model')
        self.output_size = output_size

    def forward(self, x):
        output = (ctypes.c_float * self.output_size)()
        _lib.forward(self._model, x, output)
        return output

    def free(self):
        if self._model:
            _lib.free_model(self._model)
            self._model = None


class Dataset:
    def __init__(self):
        self._dataset = _lib.create_dataset()
        self.size = 0
        self.input_size = 0
        self.output_size = 0
        self.token_count = 0
        self.vocab_size = 0

    def _refresh(self):
        self.size = _lib.dataset_size(self._dataset)
        self.input_size = _lib.dataset_input_size(self._dataset)
        self.output_size = _lib.dataset_output_size(self._dataset)
        self.token_count = _lib.dataset_token_count(self._dataset)
        self.vocab_size = _lib.dataset_vocab_size(self._dataset)

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

        success = _lib.load_csv_dataset(
            self._dataset,
            str(path).encode(),
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
        if not _lib.load_text_dataset(self._dataset, str(path).encode(), tokenizer):
            raise RuntimeError(f'Failed to load text dataset: {path}')
        self._refresh()
        return self

    def encode(self, text):
        if self.input_size <= 0:
            raise ValueError('encode() requires a table dataset with a text X column')
        output = (ctypes.c_float * self.input_size)()
        _lib.dataset_encode_text(self._dataset, text.encode(), output)
        return output

    def label(self, values):
        index = _lib.dataset_argmax(self._dataset, values)
        label = _lib.dataset_label(self._dataset, index)
        return label.decode() if label else str(index)

    def free(self):
        if self._dataset:
            _lib.free_dataset(self._dataset)
            self._dataset = None


class Sampler:
    def __init__(self, sampler):
        if not sampler:
            raise RuntimeError('Failed to create sampler')
        self._sampler = sampler

    def free(self):
        if self._sampler:
            _lib.free_sampler(self._sampler)
            self._sampler = None


class SequentialSampler(Sampler):
    def __init__(self, dataset, seed=1):
        super().__init__(_lib.create_row_sampler(dataset._dataset, SEQUENTIAL, seed))


class RandomSampler(Sampler):
    def __init__(self, dataset, replacement=False, seed=1):
        strategy = RANDOM if replacement else SHUFFLED
        super().__init__(_lib.create_row_sampler(dataset._dataset, strategy, seed))


class TokenSampler(Sampler):
    def __init__(self, dataset, sequence_length, strategy=SHUFFLED, samples=0, seed=1):
        super().__init__(_lib.create_token_sampler(
            dataset._dataset,
            sequence_length,
            strategy,
            samples,
            seed,
        ))


class DataLoader:
    def __init__(self, dataset, sampler, batch_size=1, drop_last=False):
        self._loader = _lib.create_dataloader(
            dataset._dataset,
            sampler._sampler,
            batch_size,
            int(drop_last),
        )
        if not self._loader:
            raise RuntimeError('Failed to create data loader')
        self.input_size = _lib.dataloader_input_size(self._loader)
        self.output_size = _lib.dataloader_output_size(self._loader)

    def free(self):
        if self._loader:
            _lib.free_dataloader(self._loader)
            self._loader = None


def train(model, loader, epochs, learning_rate):
    if not _lib.train_loader(model._model, loader._loader, epochs, learning_rate):
        raise RuntimeError('Training failed: model and loader shapes may not match')
