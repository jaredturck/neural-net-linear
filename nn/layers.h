#ifndef LAYERS_H
#define LAYERS_H

#include <stddef.h>
#include "activation.h"

typedef struct {
    const float* x_array;
    float* output;
    float* logits;
} BackpropCache;

typedef struct {
    int input_neurons;
    int output_neurons;
    size_t parameter_count;
    float* weights;
    float* bias;
    float* bias_gradients;
    BackpropCache* backprop_cache;
    float* gradients;
    ActivationFunction* activation_function;
    ActivationType activation_type;
    float* deltas;
    float* layer_deltas;
} Layer;

Layer* create_layer(
    int input_neurons,
    int output_neurons,
    ActivationFunction* activation_function,
    ActivationType activation_type
);
void zero_layer_gradients(Layer* layer);
void scale_layer_gradients(Layer* layer, float scale);
void free_layer(Layer* layer);
float* Linear(Layer* layer, const float* x_array);

#endif
