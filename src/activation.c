#include <math.h>

const float LAMBDA = 1.0507;
const float ALPHA = 1.67326;
const float ROOT2 = 1.4142135623730951;

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

float* relu(float x_array[], int array_size) {
    // Applies Rectified Linear Unit (ReLU) - https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
    for (int i=0; i<array_size; i++) {
        x_array[i] = (x_array[i] > 0) ? x_array[i] : 0;
    }
    return x_array;
}

float* sigmoid(float x_array[], int array_size) {
    // Applies Sigmoid activation function - https://en.wikipedia.org/wiki/Sigmoid_function
    for (int i=0; i<array_size; i++) {
        x_array[i] = 1 / (1 + expf(-x_array[i]));
    }
    return x_array;
}

float* selu(float x_array[], int array_size){
    // Applies the Scaled Exponential Linear Unit (SELU) - https://www.geeksforgeeks.org/deep-learning/selu-activation-function-in-neural-network/
    for (int i=0; i<array_size; i++) {
        if (x_array[i] > 0) {
            x_array[i] = LAMBDA * x_array[i];
        } else {
            x_array[i] = LAMBDA * ALPHA * (expf(x_array[i]) - 1);
        }
    }
    return x_array;
}

float* gelu(float x_array[], int array_size){
    // Applies the Gaussian Error Linear Unit (GELU) activation function - https://arxiv.org/pdf/1606.08415
    for (int i=0; i<array_size; i++) {
        x_array[i] = 0.5 * x_array[i] * (1 + erff(x_array[i] / ROOT2));
    }
    return x_array;
}

float* array_tanh(float x_array[], int array_size){
    // Applies Hyperbolic Tangent (tanh) - https://www.geeksforgeeks.org/deep-learning/tanh-activation-in-neural-network/`
    for (int i=0; i<array_size; i++) {
        x_array[i] = tanhf(x_array[i]);
    }
    return x_array;
}

float* softplus(float x_array[], int array_size){
    // Applies the Softplus activation function - https://www.geeksforgeeks.org/deep-learning/softplus-function-in-neural-network/
    for (int i=0; i<array_size; i++) {
        x_array[i] = logf(1 + expf(x_array[i]));
    }
    return x_array;
}

float* softmax(float x_array[], int array_size){
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
