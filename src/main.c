#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "activation.h"
#include "layers.h"
#include "backprop.h"

void display_array(float* array, int array_size) {
    for (int i=0; i<array_size; i++) {
        printf("%f, ", array[i]);
    }
    printf("\n");
}

void forward(float* x_array, Layer* layer_1, Layer* layer_2, Layer* layer_3) {
    Linear(layer_1, x_array);
    Linear(layer_2, layer_1->backprop_cache->logits);
    Linear(layer_3, layer_2->backprop_cache->logits);
}

void backward(float* y_array, Layer* layer_1, Layer* layer_2, Layer* layer_3) {
    compute_softmax_gradients(layer_3, y_array);
    compute_relu_gradients(layer_2, layer_3->layer_deltas);
    compute_relu_gradients(layer_1, layer_2->layer_deltas);
}

// Module - build 3 layer neural network
int main() {
    srand((unsigned) time(NULL));
    Layer* layer_1 = create_layer(18, 32, relu, F_RELU);
    Layer* layer_2 = create_layer(32, 32, relu, F_RELU);
    Layer* layer_3 = create_layer(32, 12, NULL, F_SOFTMAX);

    float x_array[18] = {0.47, 0.75, 0.89, 0.92, 0.72, 0.32, 0.72, 0.55, 0.5, 0.18, 0.56, 0.41, 0.92, 0.67, 0.48, 0.38, 0.7, 0.26};
    float y_array[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};

    forward(x_array, layer_1, layer_2, layer_3);
    backward(y_array, layer_1, layer_2, layer_3);
    display_array(layer_3->backprop_cache->logits, 12);
}
