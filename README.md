A neural network built completely from scratch in C, with a small Python wrapper for training and inference.

The public C API is intentionally small and sequential:

```c
Model* model = create_model(18);

add_linear(model, 32, F_RELU);
add_linear(model, 32, F_RELU);
add_linear(model, 12, F_SOFTMAX);

train(model, train_x, train_y, dataset_size, 5000, 0.001);
forward(model, input, output);

free_model(model);
```

Layer input sizes are inferred from the previous layer, so models can be built with different widths and depths without changing the training code.

Build the C library from `nn/`, then run:

```bash
python main.py
```
