Compile with gcc:

```bash
mkdir -p bin
gcc -std=c11 -O2 activation.c backprop.c dataloader.c datasets.c layers.c loss.c nn.c optimizers.c samplers.c main.c -lm -o bin/main
gcc -std=c11 -O2 -fPIC -shared activation.c backprop.c dataloader.c datasets.c layers.c loss.c nn.c optimizers.c samplers.c -lm -o bin/main.so
```

The data pipeline is split into three responsibilities:

- `Dataset`: parses table or raw-text sources and owns encoded data.
- `Sampler`: selects rows or token offsets in sequential, shuffled, or random order.
- `DataLoader`: materializes sampled items into contiguous mini-batches.

The dense trainer consumes `DataLoader` batches and accumulates gradients across the full mini-batch before one SGD update.

The CPU implementation depends only on the ISO C standard library. Neural-network, parsing, sampling, batching, and optimization algorithms are implemented in this project.
