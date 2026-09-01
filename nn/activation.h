#ifndef ACTIVATION_H
#define ACTIVATION_H

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

ActivationFunction* get_activation_function(ActivationType activation_type);
float activation_derivative(ActivationType activation_type, float x);
float relu(float x);
float sigmoid(float x);
float selu(float x);
float gelu(float x);
float array_tanh(float x);
float softplus(float x);
float* softmax(float x_array[], int array_size);

#endif
