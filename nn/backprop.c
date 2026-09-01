#include "activation.h"
#include "layers.h"

void compute_softmax_gradients(Layer* layer, float* y_array) {
    // Compute gradients for softmax activation with cross entropy loss
    for (int i=0; i<layer->output_neurons; i++) {
        layer->deltas[i] = layer->backprop_cache->logits[i] - y_array[i];
        layer->bias_gradients[i] = layer->deltas[i];
    }

    for (int i=0; i<layer->output_neurons; i++) {
        for (int j=0; j<layer->input_neurons; j++) {
            layer->gradients[i][j] = layer->deltas[i] * layer->backprop_cache->x_array[j];
        }
    }

    for (int i=0; i<layer->input_neurons; i++) {
        float total = 0.0;
        for (int j=0; j<layer->output_neurons; j++) {
            total += layer->weights[j][i] * layer->deltas[j];
        }
        layer->layer_deltas[i] = total;
    }
}

void compute_layer_gradients(Layer* layer, float* deltas) {
    for (int i=0; i<layer->output_neurons; i++) {
        float derivative = activation_derivative(layer->activation_type, layer->backprop_cache->output[i]);
        layer->deltas[i] = deltas[i] * derivative;
        layer->bias_gradients[i] = layer->deltas[i];
    }

    for (int i=0; i<layer->output_neurons; i++) {
        for (int j=0; j<layer->input_neurons; j++) {
            layer->gradients[i][j] = layer->deltas[i] * layer->backprop_cache->x_array[j];
        }
    }

    for (int i=0; i<layer->input_neurons; i++) {
        float total = 0.0;
        for (int j=0; j<layer->output_neurons; j++) {
            total += layer->weights[j][i] * layer->deltas[j];
        }
        layer->layer_deltas[i] = total;
    }
}
