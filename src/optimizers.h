#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "layers.h"

void SGD_layer(Layer* layer, float learning_rate);
void SGD(Layer* layer_1, Layer* layer_2, Layer* layer_3, float learning_rate);

#endif
