#include <math.h>

const float LAMBDA = 1.0507;
const float ALPHA = 1.67326;

void relu(float x_array[], int array_size) {
    // Applies Rectified Linear Unit (ReLU) - https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
    for (int i=0; i<array_size; i++) {
        x_array[i] = (x_array[i] > 0) ? x_array[i] : 0;
    }
}

void sigmoid(float x_array[], int array_size) {
    // Applies Sigmoid activation function - https://en.wikipedia.org/wiki/Sigmoid_function
    for (int i=0; i<array_size; i++) {
        x_array[i] = 1 / (1 + expf(-x_array[i]));
    }
}

void selu(float x_array[], int array_size){
    // Applies the Scaled Exponential Linear Unit (SELU) - https://www.geeksforgeeks.org/deep-learning/selu-activation-function-in-neural-network/
    for (int i=0; i<array_size; i++) {
        if (x_array[i] > 0) {
            x_array[i] = LAMBDA * x_array[i];
        } else {
            x_array[i] = LAMBDA * ALPHA * (expf(x_array[i]) - 1);
        }
    }
}

void gelu(){}

void tanh(){}

void softplus(){}

void softmax(){}

