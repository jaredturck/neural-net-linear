#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "activation.h"
#include "layers.h"
#include "backprop.h"
#include "loss.h"
#include "optimizers.h"

// Global layers
static Layer* g_layer_1 = NULL;
static Layer* g_layer_2 = NULL;
static Layer* g_layer_3 = NULL;
static int g_input_size = 0;
static int g_output_size = 0;

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

void train(Layer* layer_1, Layer* layer_2, Layer* layer_3, float* train_x, float* train_y, int dataset_size, int input_size, int output_size) {

    for (int epoch=0; epoch<10000; epoch++) {
        float avg_loss = 0.0;
        for (int i=0; i<dataset_size; i++) {

            float* x = train_x + i * input_size;
            float* y = train_y + i * output_size;

            forward(x, layer_1, layer_2, layer_3);
            avg_loss += categorical_cross_entropy(y, layer_3->backprop_cache->logits, output_size);

            backward(y, layer_1, layer_2, layer_3);
            SGD(layer_1, layer_2, layer_3, 0.001);
        }
        avg_loss /= dataset_size;
        if (epoch % 100 == 0) {
            printf("Epoch %d, loss %f\n", epoch+1, avg_loss);
            // display_array(layer_3->backprop_cache->logits, output_size);
        }
        if (avg_loss <= 0.05) {
            printf("Training complete at epoch %d, loss %f\n", epoch+1, avg_loss);
            return;
        }
    }
}

void init_model(int input_size, int output_size) {
    srand((unsigned) time(NULL));
    g_input_size = input_size;
    g_output_size = output_size;

    g_layer_1 = create_layer(input_size, 32, relu, F_RELU);
    g_layer_2 = create_layer(32, 32, relu, F_RELU);
    g_layer_3 = create_layer(32, output_size, NULL, F_SOFTMAX);
}

void py_train(float* train_x, float* train_y, int dataset_size, int input_size, int output_size) {
    init_model(input_size, output_size);
    train(g_layer_1, g_layer_2, g_layer_3, train_x, train_y, dataset_size, input_size, output_size);
}

void py_predict(float* x_array, float* output) {
    forward(x_array, g_layer_1, g_layer_2, g_layer_3);
    for (int i=0; i<g_output_size; i++) {
        output[i] = g_layer_3->backprop_cache->logits[i];
    }
}

int main() {
    return 0;
}
