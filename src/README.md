compile the C code with gcc
```
gcc -Ofast activation.c backprop.c datasets.c layers.c loss.c optimizers.c main.c -lm -o bin/main
gcc -shared -o libmynn.so activation.c backprop.c datasets.c layers.c loss.c optimizers.c main.c -lm -o bin/main.so
```
