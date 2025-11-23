#include "layers.h"

void SGD_layer(Layer* layer, float learning_rate) {
    for (int k=0; k<layer->output_neurons; k++) {
        for (int j=0; j<layer->input_neurons; j++) {
            layer->weights[k][j] -= learning_rate * layer-> gradients[k][j];
        }
        layer->bias[k] -= learning_rate * layer->bias_gradients[k];
    }
}

void SGD(Layer* layer_1, Layer* layer_2, Layer* layer_3, float learning_rate) {
    SGD_layer(layer_1, learning_rate); // Layer 1
    SGD_layer(layer_2, learning_rate); // Layer 2
    SGD_layer(layer_3, learning_rate); // Layer 3
}
