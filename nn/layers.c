#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "layers.h"
#include "activation.h"

Layer* create_layer(
    int input_neurons,
    int output_neurons,
    ActivationFunction* activation_function,
    ActivationType activation_type
) {
    if (input_neurons <= 0 || output_neurons <= 0) {
        return NULL;
    }

    Layer* layer = calloc(1, sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->input_neurons = input_neurons;
    layer->output_neurons = output_neurons;
    layer->parameter_count = (size_t)input_neurons * (size_t)output_neurons;
    layer->activation_function = activation_function;
    layer->activation_type = activation_type;

    layer->weights = malloc(layer->parameter_count * sizeof(float));
    layer->gradients = calloc(layer->parameter_count, sizeof(float));
    layer->bias = calloc((size_t)output_neurons, sizeof(float));
    layer->bias_gradients = calloc((size_t)output_neurons, sizeof(float));
    layer->deltas = malloc((size_t)output_neurons * sizeof(float));
    layer->layer_deltas = malloc((size_t)input_neurons * sizeof(float));
    layer->backprop_cache = calloc(1, sizeof(BackpropCache));

    if (layer->weights == NULL || layer->gradients == NULL || layer->bias == NULL ||
        layer->bias_gradients == NULL || layer->deltas == NULL || layer->layer_deltas == NULL ||
        layer->backprop_cache == NULL) {
        free_layer(layer);
        return NULL;
    }

    layer->backprop_cache->output = malloc((size_t)output_neurons * sizeof(float));
    layer->backprop_cache->logits = malloc((size_t)output_neurons * sizeof(float));
    if (layer->backprop_cache->output == NULL || layer->backprop_cache->logits == NULL) {
        free_layer(layer);
        return NULL;
    }

    const float limit = activation_type == F_RELU
        ? sqrtf(6.0f / (float)input_neurons)
        : sqrtf(6.0f / (float)(input_neurons + output_neurons));

    for (size_t index = 0; index < layer->parameter_count; index++) {
        const float unit = (float)rand() / (float)RAND_MAX;
        layer->weights[index] = (2.0f * unit - 1.0f) * limit;
    }

    return layer;
}

void zero_layer_gradients(Layer* layer) {
    memset(layer->gradients, 0, layer->parameter_count * sizeof(float));
    memset(layer->bias_gradients, 0, (size_t)layer->output_neurons * sizeof(float));
}

void scale_layer_gradients(Layer* layer, float scale) {
    for (size_t index = 0; index < layer->parameter_count; index++) {
        layer->gradients[index] *= scale;
    }
    for (int output = 0; output < layer->output_neurons; output++) {
        layer->bias_gradients[output] *= scale;
    }
}

void free_layer(Layer* layer) {
    if (layer == NULL) {
        return;
    }

    free(layer->weights);
    free(layer->bias);
    free(layer->bias_gradients);
    if (layer->backprop_cache != NULL) {
        free(layer->backprop_cache->output);
        free(layer->backprop_cache->logits);
    }
    free(layer->backprop_cache);
    free(layer->gradients);
    free(layer->deltas);
    free(layer->layer_deltas);
    free(layer);
}

float* Linear(Layer* layer, const float* x_array) {
    layer->backprop_cache->x_array = x_array;

    float* restrict preactivation = layer->backprop_cache->output;
    float* restrict activated = layer->backprop_cache->logits;
    const float* restrict weights = layer->weights;
    const int input_neurons = layer->input_neurons;

    for (int output = 0; output < layer->output_neurons; output++) {
        const float* restrict row = weights + (size_t)output * (size_t)input_neurons;
        float value = layer->bias[output];
        for (int input = 0; input < input_neurons; input++) {
            value += x_array[input] * row[input];
        }
        preactivation[output] = value;
        activated[output] = layer->activation_type == F_SOFTMAX
            ? value
            : layer->activation_function(value);
    }

    if (layer->activation_type == F_SOFTMAX) {
        softmax(activated, layer->output_neurons);
    }

    return activated;
}
