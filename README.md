A neural network built completely from scratch in C, with a small Python wrapper for model configuration, training, and inference.

All neural-network computation, dataset parsing, sampling, batching, and training logic is implemented in C using only the ISO C standard library. The Python wrapper hides `ctypes` and exposes the public framework API.

## Python API

User code only needs one framework import:

```python
import neuralnet as nn
```

Framework constants are grouped under namespaces instead of imported individually:

```python
nn.Activation.RELU
nn.Activation.SOFTMAX

nn.FieldType.FLOAT
nn.FieldType.INTEGER
nn.FieldType.TEXT
nn.FieldType.LABEL

nn.Tokenizer.BYTE

nn.SamplingStrategy.SEQUENTIAL
nn.SamplingStrategy.SHUFFLED
nn.SamplingStrategy.RANDOM
```

## Model API

Models are explicit collections of layers:

```python
class AnimalModel(nn.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add_linear(input_size, 16, nn.Activation.RELU)
        self.add_linear(16, output_size, nn.Activation.SOFTMAX)
```

## Table datasets

CSV-style files can choose any X/Y columns and describe their field types. Text widths are inferred automatically unless a fixed width is supplied.

```python
dataset = nn.Dataset().csv(
    'train.txt',
    x=[0],
    y=[1],
    types={
        0: nn.FieldType.TEXT,
        1: nn.FieldType.LABEL,
    },
)

sampler = nn.RandomSampler(dataset, seed=42)
loader = nn.DataLoader(dataset, sampler, batch_size=16)

model = AnimalModel(loader.input_size, loader.output_size)
nn.train(model, loader, 5000, 0.001)
```

Selected fields may be `FLOAT`, `INTEGER`, `TEXT`, or `LABEL`. Multiple X or Y columns are supported, along with configurable delimiters, headers, and optional fixed widths for text fields.

## Raw text datasets

Raw text is stored as a token stream. Sampling is separate from parsing, so next-token windows do not need to be materialized or advanced one token at a time.

```python
corpus = nn.Dataset().text(
    'corpus.txt',
    tokenizer=nn.Tokenizer.BYTE,
)

sampler = nn.TokenSampler(
    corpus,
    sequence_length=128,
    strategy=nn.SamplingStrategy.SHUFFLED,
    seed=42,
)

loader = nn.DataLoader(corpus, sampler, batch_size=16)
```

`TokenSampler` supports sequential non-overlapping blocks, shuffled blocks, and random offsets with replacement. In every token window, X is the sampled sequence and Y is the same sequence shifted by one token for next-token prediction.

The current dense-network trainer still uses softmax + categorical cross entropy. The raw-text loader is intended to feed the transformer/embedding training path as those model components are added.

Build the C library from `nn/`, then run:

```bash
python main.py
```
