import ctypes
from pathlib import Path


class Activation:
    RELU = 0
    SIGMOID = 1
    SELU = 2
    GELU = 3
    TANH = 4
    SOFTPLUS = 5
    SOFTMAX = 6


class FieldType:
    FLOAT = 0
    INTEGER = 1
    TEXT = 2
    LABEL = 3


class SamplingStrategy:
    SEQUENTIAL = 0
    SHUFFLED = 1
    RANDOM = 2


class _GPTTrainConfig(ctypes.Structure):
    _fields_ = [
        ('epochs', ctypes.c_int),
        ('batch_size', ctypes.c_int),
        ('steps_per_epoch', ctypes.c_int),
        ('log_every', ctypes.c_int),
        ('warmup_steps', ctypes.c_int),
        ('learning_rate', ctypes.c_float),
        ('weight_decay', ctypes.c_float),
        ('beta1', ctypes.c_float),
        ('beta2', ctypes.c_float),
        ('epsilon', ctypes.c_float),
        ('grad_clip', ctypes.c_float),
        ('seed', ctypes.c_uint),
    ]


def _load_library():
    library_path = Path(__file__).resolve().parent / 'nn' / 'bin' / 'main.so'
    return ctypes.CDLL(str(library_path))


def _configure_library(lib):
    # Dense model API.
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

    # Dataset API.
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

    # Sampler / loader API.
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

    # BPE tokenizer API.
    lib.bpe_train_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.bpe_train_file.restype = ctypes.c_void_p
    lib.bpe_load.argtypes = [ctypes.c_char_p]
    lib.bpe_load.restype = ctypes.c_void_p
    lib.bpe_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bpe_save.restype = ctypes.c_int
    lib.bpe_vocab_size.argtypes = [ctypes.c_void_p]
    lib.bpe_vocab_size.restype = ctypes.c_int
    lib.bpe_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.bpe_encode.restype = ctypes.c_int
    lib.bpe_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.bpe_decode.restype = ctypes.c_int
    lib.free_bpe_tokenizer.argtypes = [ctypes.c_void_p]
    lib.free_bpe_tokenizer.restype = None

    # GPT API.
    lib.create_gpt_model.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.create_gpt_model.restype = ctypes.c_void_p
    lib.gpt_vocab_size.argtypes = [ctypes.c_void_p]
    lib.gpt_vocab_size.restype = ctypes.c_int
    lib.gpt_context_length.argtypes = [ctypes.c_void_p]
    lib.gpt_context_length.restype = ctypes.c_int
    lib.gpt_embedding_dim.argtypes = [ctypes.c_void_p]
    lib.gpt_embedding_dim.restype = ctypes.c_int
    lib.gpt_head_count.argtypes = [ctypes.c_void_p]
    lib.gpt_head_count.restype = ctypes.c_int
    lib.gpt_layer_count.argtypes = [ctypes.c_void_p]
    lib.gpt_layer_count.restype = ctypes.c_int
    lib.gpt_hidden_dim.argtypes = [ctypes.c_void_p]
    lib.gpt_hidden_dim.restype = ctypes.c_int
    lib.gpt_train_file.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(_GPTTrainConfig),
    ]
    lib.gpt_train_file.restype = ctypes.c_int
    lib.gpt_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.gpt_generate.restype = ctypes.c_void_p
    lib.gpt_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.gpt_save.restype = ctypes.c_int
    lib.gpt_load.argtypes = [ctypes.c_char_p]
    lib.gpt_load.restype = ctypes.c_void_p
    lib.gpt_free_bytes.argtypes = [ctypes.c_void_p]
    lib.gpt_free_bytes.restype = None
    lib.free_gpt_model.argtypes = [ctypes.c_void_p]
    lib.free_gpt_model.restype = None


_lib = _load_library()
_configure_library(_lib)


def _columns(value):
    if isinstance(value, int):
        return [value]
    return list(value)


def _int_array(values):
    return (ctypes.c_int * len(values))(*values)


class Tokenizer:
    BYTE = 0

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.vocab_size = _lib.bpe_vocab_size(tokenizer) if tokenizer else 0

    @classmethod
    def train(cls, path, vocab_size=512):
        tokenizer = _lib.bpe_train_file(str(path).encode(), vocab_size)
        if not tokenizer:
            raise RuntimeError(f'Failed to train tokenizer from: {path}')
        return cls(tokenizer)

    @classmethod
    def load(cls, path):
        tokenizer = _lib.bpe_load(str(path).encode())
        if not tokenizer:
            raise RuntimeError(f'Failed to load tokenizer: {path}')
        return cls(tokenizer)

    def save(self, path):
        if not self._tokenizer or not _lib.bpe_save(self._tokenizer, str(path).encode()):
            raise RuntimeError(f'Failed to save tokenizer: {path}')

    def encode(self, text):
        if not self._tokenizer:
            raise ValueError('encode() requires a trained BPE tokenizer')
        data = text.encode() if isinstance(text, str) else bytes(text)
        buffer = ctypes.create_string_buffer(data, len(data)) if data else None
        pointer = ctypes.cast(buffer, ctypes.c_void_p) if buffer is not None else None
        count = _lib.bpe_encode(self._tokenizer, pointer, len(data), None, 0)
        if count < 0:
            raise RuntimeError('Tokenization failed')
        if count == 0:
            return []
        output = (ctypes.c_int32 * count)()
        if _lib.bpe_encode(self._tokenizer, pointer, len(data), output, count) != count:
            raise RuntimeError('Tokenization failed')
        return list(output)

    def decode(self, tokens):
        if not self._tokenizer:
            raise ValueError('decode() requires a trained BPE tokenizer')
        tokens = list(tokens)
        if not tokens:
            return ''
        token_array = (ctypes.c_int32 * len(tokens))(*tokens)
        size = _lib.bpe_decode(self._tokenizer, token_array, len(tokens), None, 0)
        if size < 0:
            raise RuntimeError('Token decoding failed')
        output = ctypes.create_string_buffer(size)
        if _lib.bpe_decode(self._tokenizer, token_array, len(tokens), output, size) != size:
            raise RuntimeError('Token decoding failed')
        return bytes(output.raw[:size]).decode('utf-8', errors='replace')

    def free(self):
        if self._tokenizer:
            _lib.free_bpe_tokenizer(self._tokenizer)
            self._tokenizer = None
            self.vocab_size = 0


class AdamW:
    def __init__(
        self,
        learning_rate=3e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


class GPT:
    def __init__(
        self,
        vocab_size,
        context_length=128,
        embedding_dim=128,
        heads=4,
        layers=4,
        hidden_dim=None,
        seed=1,
    ):
        hidden_dim = hidden_dim or embedding_dim * 4
        self._gpt = _lib.create_gpt_model(
            vocab_size,
            context_length,
            embedding_dim,
            heads,
            layers,
            hidden_dim,
            seed,
        )
        if not self._gpt:
            raise ValueError(
                'Invalid GPT configuration: embedding_dim must be divisible by heads '
                'and each attention head must have an even dimension'
            )
        self._refresh()

    def _refresh(self):
        self.vocab_size = _lib.gpt_vocab_size(self._gpt)
        self.context_length = _lib.gpt_context_length(self._gpt)
        self.embedding_dim = _lib.gpt_embedding_dim(self._gpt)
        self.heads = _lib.gpt_head_count(self._gpt)
        self.layers = _lib.gpt_layer_count(self._gpt)
        self.hidden_dim = _lib.gpt_hidden_dim(self._gpt)

    @classmethod
    def load(cls, path):
        pointer = _lib.gpt_load(str(path).encode())
        if not pointer:
            raise RuntimeError(f'Failed to load GPT model: {path}')
        model = cls.__new__(cls)
        model._gpt = pointer
        model._refresh()
        return model

    def train(
        self,
        tokenizer,
        path,
        epochs=1,
        batch_size=16,
        optimizer=None,
        steps_per_epoch=0,
        warmup_steps=0,
        grad_clip=1.0,
        seed=1,
        log_every=10,
    ):
        if not tokenizer._tokenizer:
            raise ValueError('GPT training requires a trained BPE tokenizer')
        if tokenizer.vocab_size != self.vocab_size:
            raise ValueError('Tokenizer vocabulary does not match model vocabulary')
        optimizer = optimizer or AdamW()
        config = _GPTTrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            log_every=log_every,
            warmup_steps=warmup_steps,
            learning_rate=optimizer.learning_rate,
            weight_decay=optimizer.weight_decay,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            epsilon=optimizer.epsilon,
            grad_clip=grad_clip,
            seed=seed,
        )
        if not _lib.gpt_train_file(
            self._gpt,
            tokenizer._tokenizer,
            str(path).encode(),
            ctypes.byref(config),
        ):
            raise RuntimeError('GPT training failed')

    def generate(
        self,
        tokenizer,
        prompt,
        max_tokens=100,
        temperature=0.8,
        top_k=40,
        seed=1,
    ):
        if not tokenizer._tokenizer:
            raise ValueError('Generation requires a trained BPE tokenizer')
        data = prompt.encode() if isinstance(prompt, str) else bytes(prompt)
        if not data:
            raise ValueError('Prompt cannot be empty')
        buffer = ctypes.create_string_buffer(data, len(data))
        output_length = ctypes.c_int()
        result = _lib.gpt_generate(
            self._gpt,
            tokenizer._tokenizer,
            ctypes.cast(buffer, ctypes.c_void_p),
            len(data),
            max_tokens,
            temperature,
            top_k,
            seed,
            ctypes.byref(output_length),
        )
        if not result:
            raise RuntimeError('Generation failed')
        try:
            output = ctypes.string_at(result, output_length.value)
        finally:
            _lib.gpt_free_bytes(result)
        return output.decode('utf-8', errors='replace')

    def save(self, path):
        if not _lib.gpt_save(self._gpt, str(path).encode()):
            raise RuntimeError(f'Failed to save GPT model: {path}')

    def free(self):
        if self._gpt:
            _lib.free_gpt_model(self._gpt)
            self._gpt = None


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
        x_widths = [widths.get(column, 0 if types[column] == FieldType.TEXT else 1) for column in x]
        y_widths = [widths.get(column, 0 if types[column] == FieldType.TEXT else 1) for column in y]
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

    def text(self, path, tokenizer=Tokenizer.BYTE):
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
        super().__init__(_lib.create_row_sampler(dataset._dataset, SamplingStrategy.SEQUENTIAL, seed))


class RandomSampler(Sampler):
    def __init__(self, dataset, replacement=False, seed=1):
        strategy = SamplingStrategy.RANDOM if replacement else SamplingStrategy.SHUFFLED
        super().__init__(_lib.create_row_sampler(dataset._dataset, strategy, seed))


class TokenSampler(Sampler):
    def __init__(self, dataset, sequence_length, strategy=SamplingStrategy.SHUFFLED, samples=0, seed=1):
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


# Backward-compatible flat aliases.
F_RELU = Activation.RELU
F_SIGMOID = Activation.SIGMOID
F_SELU = Activation.SELU
F_GELU = Activation.GELU
F_TANH = Activation.TANH
F_SOFTPLUS = Activation.SOFTPLUS
F_SOFTMAX = Activation.SOFTMAX
FLOAT = FieldType.FLOAT
INTEGER = FieldType.INTEGER
TEXT = FieldType.TEXT
LABEL = FieldType.LABEL
BYTE = Tokenizer.BYTE
SEQUENTIAL = SamplingStrategy.SEQUENTIAL
SHUFFLED = SamplingStrategy.SHUFFLED
RANDOM = SamplingStrategy.RANDOM

__all__ = [
    'Activation',
    'FieldType',
    'Tokenizer',
    'SamplingStrategy',
    'AdamW',
    'GPT',
    'Model',
    'Dataset',
    'Sampler',
    'SequentialSampler',
    'RandomSampler',
    'TokenSampler',
    'DataLoader',
    'train',
]
