#include "activation.h"
#include <math.h>
#include <stddef.h>

const float LAMBDA = 1.0507;
const float ALPHA = 1.67326;
const float ROOT2 = 1.4142135623730951;

ActivationFunction* get_activation_function(ActivationType activation_type) {
    switch (activation_type) {
        case F_RELU:
            return relu;
        case F_SIGMOID:
            return sigmoid;
        case F_SELU:
            return selu;
        case F_GELU:
            return gelu;
        case F_TANH:
            return array_tanh;
        case F_SOFTPLUS:
            return softplus;
        case F_SOFTMAX:
            return NULL;
    }
    return NULL;
}

float activation_derivative(ActivationType activation_type, float x) {
    switch (activation_type) {
        case F_RELU:
            return (x > 0) ? 1.0 : 0.0;
        case F_SIGMOID: {
            float y = sigmoid(x);
            return y * (1 - y);
        }
        case F_SELU:
            return (x > 0) ? LAMBDA : LAMBDA * ALPHA * expf(x);
        case F_GELU:
            return 0.5 * (1 + erff(x / ROOT2)) + (x * expf(-0.5 * x * x) / 2.5066282746310002);
        case F_TANH: {
            float y = tanhf(x);
            return 1 - y * y;
        }
        case F_SOFTPLUS:
            return sigmoid(x);
        case F_SOFTMAX:
            return 0.0;
    }
    return 0.0;
}

float array_max(float x_array[], int array_size) {
    // Return largest number in an array
    float largest = x_array[0];
    for (int i=1; i<array_size; i++) {
        if (x_array[i] > largest) {
            largest = x_array[i];
        }
    }
    return largest;
}

float relu(float x) {
    // Applies Rectified Linear Unit (ReLU) - https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
    return (x > 0) ? x : 0;
}

float sigmoid(float x) {
    // Applies Sigmoid activation function - https://en.wikipedia.org/wiki/Sigmoid_function
    return 1 / (1 + expf(-x));
}

float selu(float x) {
    // Applies the Scaled Exponential Linear Unit (SELU) - https://www.geeksforgeeks.org/deep-learning/selu-activation-function-in-neural-network/
    if (x > 0) {
        return LAMBDA * x;
    } else {
        return LAMBDA * ALPHA * (expf(x) - 1);
    }
}

float gelu(float x) {
    // Applies the Gaussian Error Linear Unit (GELU) activation function - https://arxiv.org/pdf/1606.08415
    return 0.5 * x * (1 + erff(x / ROOT2));
}

float array_tanh(float x) {
    // Applies Hyperbolic Tangent (tanh) - https://www.geeksforgeeks.org/deep-learning/tanh-activation-in-neural-network/`
    return tanhf(x);
}

float softplus(float x) {
    // Applies the Softplus activation function - https://www.geeksforgeeks.org/deep-learning/softplus-function-in-neural-network/
    return logf(1 + expf(x));
}

float* softmax(float x_array[], int array_size) {
    // Applies Softmax - https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/
    float max_x = array_max(x_array, array_size);
    float total = 0.0;

    for (int i=0; i<array_size; i++) {
        x_array[i] = expf(x_array[i] - max_x);
        total += x_array[i];
    }
    for (int i=0; i<array_size; i++) {
        x_array[i] = x_array[i] / total;
    }
    return x_array;
}
