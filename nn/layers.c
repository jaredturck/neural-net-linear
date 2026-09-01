#include <math.h>
#include "layers.h"
#include "activation.h"

Layer* create_layer(
    int input_neurons,
    int output_neurons,
    ActivationFunction* activation_function,
    ActivationType activation_type
) {
    Layer* layer = malloc(sizeof(Layer));
    layer->input_neurons = input_neurons;
    layer->output_neurons = output_neurons;

    layer->weights = malloc((size_t)output_neurons * sizeof(float*));
    float limit = activation_type == F_RELU
        ? sqrtf(6.0f / (float)input_neurons)
        : sqrtf(6.0f / (float)(input_neurons + output_neurons));

    for (int i = 0; i < output_neurons; i++) {
        layer->weights[i] = malloc((size_t)input_neurons * sizeof(float));
        for (int j = 0; j < input_neurons; j++) {
            float unit = (float)rand() / (float)RAND_MAX;
            layer->weights[i][j] = (2.0f * unit - 1.0f) * limit;
        }
    }

    layer->bias = calloc((size_t)output_neurons, sizeof(float));
    layer->bias_gradients = calloc((size_t)output_neurons, sizeof(float));

    layer->backprop_cache = malloc(sizeof(BackpropCache));
    layer->backprop_cache->x_array = NULL;
    layer->backprop_cache->output = calloc((size_t)output_neurons, sizeof(float));
    layer->backprop_cache->logits = calloc((size_t)output_neurons, sizeof(float));

    layer->gradients = malloc((size_t)output_neurons * sizeof(float*));
    for (int i = 0; i < output_neurons; i++) {
        layer->gradients[i] = calloc((size_t)input_neurons, sizeof(float));
    }

    layer->activation_function = activation_function;
    layer->activation_type = activation_type;
    layer->deltas = malloc((size_t)output_neurons * sizeof(float));
    layer->layer_deltas = malloc((size_t)input_neurons * sizeof(float));

    return layer;
}

void zero_layer_gradients(Layer* layer) {
    for (int i = 0; i < layer->output_neurons; i++) {
        layer->bias_gradients[i] = 0.0f;
        for (int j = 0; j < layer->input_neurons; j++) {
            layer->gradients[i][j] = 0.0f;
        }
    }
}

void scale_layer_gradients(Layer* layer, float scale) {
    for (int i = 0; i < layer->output_neurons; i++) {
        layer->bias_gradients[i] *= scale;
        for (int j = 0; j < layer->input_neurons; j++) {
            layer->gradients[i][j] *= scale;
        }
    }
}

void free_layer(Layer* layer) {
    for (int i = 0; i < layer->output_neurons; i++) {
        free(layer->weights[i]);
        free(layer->gradients[i]);
    }

    free(layer->weights);
    free(layer->bias);
    free(layer->bias_gradients);
    free(layer->backprop_cache->output);
    free(layer->backprop_cache->logits);
    free(layer->backprop_cache);
    free(layer->gradients);
    free(layer->deltas);
    free(layer->layer_deltas);
    free(layer);
}

float* Linear(Layer* layer, float* x_array) {
    layer->backprop_cache->x_array = x_array;

    if (layer->activation_type == F_SOFTMAX) {
        for (int i = 0; i < layer->output_neurons; i++) {
            float y = 0.0f;
            for (int j = 0; j < layer->input_neurons; j++) {
                y += x_array[j] * layer->weights[i][j];
            }
            y += layer->bias[i];
            layer->backprop_cache->output[i] = y;
            layer->backprop_cache->logits[i] = y;
        }
        softmax(layer->backprop_cache->logits, layer->output_neurons);
    } else {
        for (int i = 0; i < layer->output_neurons; i++) {
            float y = 0.0f;
            for (int j = 0; j < layer->input_neurons; j++) {
                y += x_array[j] * layer->weights[i][j];
            }
            y += layer->bias[i];
            layer->backprop_cache->output[i] = y;
            layer->backprop_cache->logits[i] = layer->activation_function(y);
        }
    }
    return layer->backprop_cache->logits;
}
