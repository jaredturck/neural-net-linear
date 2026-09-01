A neural network built completely from scratch in C, with a small Python wrapper for training and inference.

The Python entry point keeps model structure explicit while all computation stays in C:

```python
class AnimalModel:
    def __init__(self, input_size, output_size):
        self.model = Model()
        self.model.add_linear(input_size, 16, F_RELU)
        self.model.add_linear(16, output_size, F_SOFTMAX)
```

Datasets use the same explicit style:

```python
dataset = Dataset()
dataset.load_animal('train.txt')

model = AnimalModel(dataset.input_size, dataset.output_size)
train(model.model, dataset, 5000, 0.001)
```

Build the C library from `nn/`, then run:

```bash
python main.py
```
