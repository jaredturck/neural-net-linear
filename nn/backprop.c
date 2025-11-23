#include "layers.h"

void compute_softmax_gradients(Layer* layer, float* y_array) {
    // Compute graidents for softmax activation with cross entropy loss

    // Deltas
    for (int i=0; i<layer->output_neurons; i++) {
        layer->deltas[i] = layer->backprop_cache->logits[i] - y_array[i];
    }

    // Compute gradients
    for (int i=0; i<layer->output_neurons; i++) {
        for (int j=0; j<layer->input_neurons; j++) {
            layer->gradients[i][j] = layer->deltas[i] * layer->backprop_cache->x_array[j];
        }
    }
    layer->bias_gradients = layer->deltas;

    for (int i=0; i<layer->input_neurons; i++) {
        float total = 0.0;
        for (int j=0; j<layer->output_neurons; j++) {
            total += layer->weights[j][i] * layer->deltas[j];
        }
        layer->layer_deltas[i] = total;
    }
}

void compute_relu_gradients(Layer* layer, float* deltas) {
    // Compute dragients for relu activation

    // calculate mask
    for (int i=0; i<layer->output_neurons; i++) {
        layer->mask[i] = (int)(layer->backprop_cache->output[i] > 0);
    }

    // Compute deltas
    for (int i=0; i<layer->output_neurons; i++) {
        layer->deltas[i] = deltas[i] * layer->mask[i];
    }

    // Compute gradients
    for (int i=0; i<layer->output_neurons; i++) {
        for (int j=0; j<layer->input_neurons; j++) {
            layer->gradients[i][j] = layer->deltas[i] * layer->backprop_cache->x_array[j];
        }
    }
    layer->bias_gradients = layer->deltas;

    for (int i=0; i<layer->input_neurons; i++) {
        float total = 0.0;
        for (int j=0; j<layer->output_neurons; j++) {
            total += layer->weights[j][i] * layer->deltas[j];
        }
        layer->layer_deltas[i] = total;
    }

}
