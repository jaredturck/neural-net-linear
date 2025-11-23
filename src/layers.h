#ifndef LAYERS_H
#define LAYERS_H

#include <stdlib.h>
typedef float (ActivationFunction)(float x);

typedef enum {
    F_RELU,
    F_SIGMOID,
    F_SELU,
    F_GELU,
    F_TANH,
    F_SOFTPLUS,
    F_SOFTMAX
} ActivationType;

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
    int* mask;
} Layer;

Layer* create_layer(int input_neurons, int output_neurons, ActivationFunction* activation_function, ActivationType activation_type);
float* Linear(Layer* layer, float* x_array);

void compute_softmax_gradients(Layer* layer, float* y_array);
void compute_relu_gradients(Layer* layer, float* y_array);

#endif
