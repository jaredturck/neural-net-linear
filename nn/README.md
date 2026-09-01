Compile with gcc:

```bash
mkdir -p bin
gcc -std=c11 -O3 activation.c backprop.c dataloader.c datasets.c layers.c loss.c nn.c optimizers.c samplers.c main.c -lm -o bin/main
gcc -std=c11 -O3 -fPIC -shared activation.c backprop.c dataloader.c datasets.c layers.c loss.c nn.c optimizers.c samplers.c -lm -o bin/main.so
```

The data pipeline is split into three responsibilities:

- `Dataset`: parses table or raw-text sources and owns encoded data.
- `Sampler`: selects rows or token offsets in sequential, shuffled, or random order.
- `DataLoader`: materializes sampled items into contiguous mini-batches.

Table `TEXT` fields are categorical features. Each text field fits a byte vocabulary from the loaded data and one-hot encodes each character position; unknown characters use a reserved category and padding positions remain zero. Raw-text datasets remain token streams for future embedding and transformer layers.

The dense trainer accumulates parameter gradients across each mini-batch, averages those gradients by the actual batch size, and then performs one SGD update using the learning rate supplied by the caller unchanged.

Dense-layer initialization is fan-aware: ReLU layers use a He-style uniform range, while other dense layers use a Xavier-style uniform range.

## CPU kernel layout

Dense-layer weights and gradients are stored as contiguous row-major buffers rather than arrays of separately allocated row pointers. Forward propagation walks each output row linearly, and backpropagation fuses parameter-gradient accumulation with propagation of input deltas so the weight matrix is traversed once in cache-friendly order.

Gradient zeroing uses bulk contiguous memory operations, while gradient scaling and SGD updates use flat loops over parameter buffers. This layout reduces allocation overhead and pointer chasing on CPU and also matches the flat device-buffer model that will be used by a future CUDA backend.

The CPU implementation depends only on the ISO C standard library. Neural-network, parsing, sampling, batching, and optimization algorithms are implemented in this project.
