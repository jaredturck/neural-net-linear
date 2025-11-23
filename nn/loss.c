#include <math.h>

float mse(float* y_true, float* y_pred, int array_size) {
    // Mean Squared Error (MSE) - https://www.geeksforgeeks.org/maths/mean-squared-error/
    float total = 0.0;
    for (int i=0; i<array_size; i++) {
        total += powf(y_true[i] - y_pred[i], 2);
    }
    return total / array_size;
}

float binary_cross_entropy(float* y_true, float* y_pred, int array_size) {
    // Binary Cross Entropy (BCE) - https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/
    float total = 0.0;
    for (int i=0; i<array_size; i++) {
        total += y_true[i] * logf(y_pred[i]) + (1 - y_true[i]) * logf(1 - y_pred[i]);
    }
    return -total / array_size;
}

float categorical_cross_entropy(float* y_true, float* y_pred, int array_size) {
    // Categorical Cross Entropy (CCE) - https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
    float total = 0.0;
    for (int i=0; i<array_size; i++) {
        total += y_true[i] * logf(y_pred[i]);
    }
    return -total;
}
