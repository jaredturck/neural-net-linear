#ifndef BACKPROP_H
#define BACKPROP_H

#include "layers.h"

void compute_softmax_gradients(Layer* layer, float* y_array);
void compute_layer_gradients(Layer* layer, float* deltas);

#endif
