#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "activation.h"
#include "layers.h"
#include "backprop.h"
#include "loss.h"
#include "optimizers.h"

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

void train(Layer* layer_1, Layer* layer_2, Layer* layer_3) {

    float train_x[5][18] = {
        {0.89, 0.7, 0.15, 0.7, 0.25, 0.32, 0.96, 1.0, 0.71, 0.66, 0.2, 0.92, 0.57, 0.81, 0.47, 0.22, 0.88, 0.75},
        {0.55, 0.4, 0.94, 0.01, 0.43, 0.97, 0.18, 0.5, 0.43, 0.51, 0.47, 0.17, 0.65, 0.27, 0.02, 0.57, 0.68, 0.55},
        {0.54, 0.81, 0.87, 0.66, 0.95, 0.64, 0.89, 0.25, 0.38, 0.55, 0.83, 0.37, 0.1, 0.02, 0.26, 0.9, 0.84, 0.88},
        {0.08, 0.26, 0.66, 0.2, 0.33, 0.99, 0.76, 0.44, 0.93, 0.98, 0.61, 0.93, 0.71, 0.05, 0.49, 0.52, 0.89, 0.9},
        {1.0, 0.02, 0.95, 0.07, 0.81, 0.68, 0.5, 0.63, 0.8, 0.05, 0.46, 0.57, 0.39, 0.68, 0.04, 0.04, 0.07, 0.73}
    };
    float train_y[5][12] = {
        {0,0,0,0,0,0,0,0,0,0,1,0},
        {0,0,0,0,0,0,0,0,0,1,0,0},
        {0,0,0,0,0,0,0,0,1,0,0,0},
        {0,0,0,0,0,0,0,1,0,0,0,0},
        {0,0,0,0,0,0,1,0,0,0,0,0}
    };
    int dataset_size = 5;
    int y_size = 12;

    for (int epoch=0; epoch<10000; epoch++) {
        float avg_loss = 0.0;
        for (int i=0; i<dataset_size; i++) {
            forward(train_x[i], layer_1, layer_2, layer_3);
            avg_loss += categorical_cross_entropy(train_y[i], layer_3->backprop_cache->logits, y_size);

            backward(train_y[i], layer_1, layer_2, layer_3);
            SGD(layer_1, layer_2, layer_3, 0.001);
        }
        avg_loss /= dataset_size;
        if (epoch % 100 == 0) {
            printf("Epoch %d, loss %f\n", epoch+1, avg_loss);
            // display_array(layer_3->backprop_cache->logits, y_size);
        }
        if (avg_loss <= 0.05) {
            printf("Training complete at epoch %d, loss %f\n", epoch+1, avg_loss);
            return;
        }
    }
}

// Module - build 3 layer neural network
int main() {
    srand((unsigned) time(NULL));
    Layer* layer_1 = create_layer(18, 32, relu, F_RELU);
    Layer* layer_2 = create_layer(32, 32, relu, F_RELU);
    Layer* layer_3 = create_layer(32, 12, NULL, F_SOFTMAX);

    train(layer_1, layer_2, layer_3);
}
