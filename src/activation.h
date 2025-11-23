#ifndef ACTIVATION_H
#define ACTIVATION_H

float relu(float x);
float sigmoid(float x);
float selu(float x);
float gelu(float x);
float array_tanh(float x);
float softplus(float x);
float* softmax(float x_array[], int array_size);

#endif
