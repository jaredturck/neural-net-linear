#include "layers.h"

void SGD_layer(Layer* layer, float learning_rate) {
    for (int i=0; i<layer->output_neurons; i++) {
        for (int j=0; j<layer->input_neurons; j++) {
            layer->weights[i][j] -= learning_rate * layer->gradients[i][j];
        }
        layer->bias[i] -= learning_rate * layer->bias_gradients[i];
    }
}
