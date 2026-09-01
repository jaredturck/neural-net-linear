#include <string.h>
#include "activation.h"
#include "layers.h"

static void accumulate_parameter_and_input_gradients(Layer* layer) {
    const int input_neurons = layer->input_neurons;
    const float* restrict x = layer->backprop_cache->x_array;
    const float* restrict weights = layer->weights;
    float* restrict gradients = layer->gradients;
    float* restrict propagated = layer->layer_deltas;

    memset(propagated, 0, (size_t)input_neurons * sizeof(float));

    for (int output = 0; output < layer->output_neurons; output++) {
        const float delta = layer->deltas[output];
        const size_t row_offset = (size_t)output * (size_t)input_neurons;
        const float* restrict weight_row = weights + row_offset;
        float* restrict gradient_row = gradients + row_offset;

        for (int input = 0; input < input_neurons; input++) {
            gradient_row[input] += delta * x[input];
            propagated[input] += weight_row[input] * delta;
        }
    }
}

void compute_softmax_gradients(Layer* layer, float* y_array) {
    for (int output = 0; output < layer->output_neurons; output++) {
        const float delta = layer->backprop_cache->logits[output] - y_array[output];
        layer->deltas[output] = delta;
        layer->bias_gradients[output] += delta;
    }

    accumulate_parameter_and_input_gradients(layer);
}

void compute_layer_gradients(Layer* layer, float* deltas) {
    for (int output = 0; output < layer->output_neurons; output++) {
        const float derivative = activation_derivative(
            layer->activation_type,
            layer->backprop_cache->output[output]
        );
        const float delta = deltas[output] * derivative;
        layer->deltas[output] = delta;
        layer->bias_gradients[output] += delta;
    }

    accumulate_parameter_and_input_gradients(layer);
}
