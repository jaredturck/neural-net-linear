A neural network built completely from scratch in C, with a small Python wrapper over the C API.

All neural-network computation, dataset parsing, sampling, batching, and training logic is implemented in C using only the ISO C standard library. `neuralnet.py` contains the `ctypes` binding and Python-facing framework classes; `main.py` is the user entry point for defining a model, configuring data, training, and inference.

## Python API

User code does not need to configure `ctypes` or call raw C functions directly:

```python
from neuralnet import (
    DataLoader,
    Dataset,
    F_RELU,
    F_SOFTMAX,
    LABEL,
    Model,
    RandomSampler,
    TEXT,
    train,
)


class AnimalModel(Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_linear(input_size, 16, F_RELU)
        self.add_linear(16, output_size, F_SOFTMAX)
```

The wrapper is only interface plumbing. Model execution, backpropagation, optimization, dataset processing, sampling, and batching remain in C.

## Table datasets

CSV-style files can choose any X/Y columns and describe their field types. Text widths are inferred automatically unless a fixed width is supplied.

```python
dataset = Dataset().csv(
    'train.txt',
    x=[0],
    y=[1],
    types={0: TEXT, 1: LABEL},
)

sampler = RandomSampler(dataset, seed=42)
loader = DataLoader(dataset, sampler, batch_size=16)

model = AnimalModel(loader.input_size, loader.output_size)
train(model, loader, 5000, 0.001)
```

Selected fields may be `FLOAT`, `INTEGER`, `TEXT`, or `LABEL`. Multiple X or Y columns are supported, along with configurable delimiters, headers, and optional fixed widths for text fields.

## Raw text datasets

Raw text is stored as a token stream. Sampling is separate from parsing, so next-token windows do not need to be materialized or advanced one token at a time.

```python
corpus = Dataset().text('corpus.txt')

sampler = TokenSampler(
    corpus,
    sequence_length=128,
    strategy=SHUFFLED,
    seed=42,
)

loader = DataLoader(corpus, sampler, batch_size=16)
```

`TokenSampler` supports sequential non-overlapping blocks, shuffled blocks, and random offsets with replacement. In every token window, X is the sampled sequence and Y is the same sequence shifted by one token for next-token prediction.

The current dense-network trainer still uses softmax + categorical cross entropy. The raw-text loader is intended to feed the transformer/embedding training path as those model components are added.

Build the C library from `nn/`, then run:

```bash
python main.py
```
