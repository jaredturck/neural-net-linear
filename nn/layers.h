#ifndef LAYERS_H
#define LAYERS_H

#include <stdlib.h>
#include "activation.h"

typedef struct {
    float* x_array;
    float* output;
    float* logits;
} BackpropCache;

typedef struct {
    int input_neurons;
    int output_neurons;
    float** weights;
    float* bias;
    float* bias_gradients;
    BackpropCache* backprop_cache;
    float** gradients;
    ActivationFunction* activation_function;
    ActivationType activation_type;
    float* deltas;
    float* layer_deltas;
} Layer;

Layer* create_layer(int input_neurons, int output_neurons, ActivationFunction* activation_function, ActivationType activation_type);
void free_layer(Layer* layer);
float* Linear(Layer* layer, float* x_array);

#endif
