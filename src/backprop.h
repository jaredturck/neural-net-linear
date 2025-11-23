#ifndef BACKPROP_H
#define BACKPROP_H

void compute_softmax_gradients(Layer* layer, float* y_array);
void compute_relu_gradients(Layer* layer, float* deltas);

#endif
