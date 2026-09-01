# neural-net-linear

A neural-network framework built from scratch in C with a small Python interface for defining, training, and running models. The CPU implementation uses only the ISO C standard library, with CUDA support planned as the next backend.

```bash
mkdir -p nn/bin

gcc -std=c11 -O3 -fPIC -shared \
    nn/activation.c \
    nn/backprop.c \
    nn/dataloader.c \
    nn/datasets.c \
    nn/layers.c \
    nn/loss.c \
    nn/nn.c \
    nn/optimizers.c \
    nn/samplers.c \
    -lm \
    -o nn/bin/main.so

python main.py
```
