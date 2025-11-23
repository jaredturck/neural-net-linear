#ifndef LOSS_H
#define LOSS_H

float mse(float* y_true, float* y_pred, int array_size);
float binary_cross_entropy(float* y_true, float* y_pred, int array_size);
float categorical_cross_entropy(float* y_true, float* y_pred, int array_size);

#endif
