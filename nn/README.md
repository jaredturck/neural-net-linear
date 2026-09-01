Compile the C code with gcc:

```bash
mkdir -p bin
gcc -O2 activation.c backprop.c datasets.c layers.c loss.c optimizers.c nn.c main.c -lm -o bin/main
gcc -O2 -fPIC -shared activation.c backprop.c datasets.c layers.c loss.c optimizers.c nn.c -lm -o bin/main.so
```

Public model API:

```c
#include "nn.h"

Model* model = create_model();
add_linear(model, 18, 32, F_RELU);
add_linear(model, 32, 12, F_SOFTMAX);

train(model, train_x, train_y, dataset_size, 5000, 0.001);
forward(model, input, output);

free_model(model);
```

Dataset preparation is also implemented in C:

```c
#include "datasets.h"

Dataset* dataset = create_dataset();
load_animal_dataset(dataset, "../train.txt");
train_dataset(model, dataset, 5000, 0.001);
free_dataset(dataset);
```

`Model` and `Dataset` are opaque. The Python entry point only loads this API with `ctypes`.
