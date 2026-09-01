#include "layers.h"

void SGD_layer(Layer* layer, float learning_rate) {
    for (size_t index = 0; index < layer->parameter_count; index++) {
        layer->weights[index] -= learning_rate * layer->gradients[index];
    }
    for (int output = 0; output < layer->output_neurons; output++) {
        layer->bias[output] -= learning_rate * layer->bias_gradients[output];
    }
}
