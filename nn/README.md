Compile the C code with gcc:

```bash
mkdir -p bin
gcc -O2 activation.c backprop.c datasets.c layers.c loss.c optimizers.c nn.c main.c -lm -o bin/main
gcc -O2 -fPIC -shared activation.c backprop.c datasets.c layers.c loss.c optimizers.c nn.c -lm -o bin/main.so
```

Public API:

```c
#include "nn.h"

Model* model = create_model(18);
add_linear(model, 32, F_RELU);
add_linear(model, 12, F_SOFTMAX);

train(model, train_x, train_y, dataset_size, 5000, 0.001);
forward(model, input, output);

free_model(model);
```

`Model` is opaque. Layer storage, weights, caches, and execution details stay internal to the library.
