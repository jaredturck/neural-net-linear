#include <stdlib.h>

typedef void (ActivationFunction)(const float* input, float* output, int length);

typedef struct {
    float* x_array;
    float* output;
    float* logits;
} BackpropCache;

typedef struct {
    int input_neurons;
    int output_neurons;
    float** weights;
    float* bias;
    float* bias_gradients;
    BackpropCache* backprop_cache;
    float** gradients;
    ActivationFunction* activation_function;
} Layer;

Layer* create_layer(int* input_neurons, int* output_neurons, ActivationFunction* activation_function) {
    Layer* layer = malloc(sizeof(Layer));
    layer->input_neurons = *input_neurons;
    layer->output_neurons = *output_neurons;

    // Initialize weights
    layer->weights = malloc(*output_neurons * sizeof(float*));
    for (int i=0; i<*output_neurons; i++) {
        layer->weights[i] = malloc(*input_neurons * sizeof(float));
        for (int j=0; j<*input_neurons; j++) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
    }

    // Initialize bias
    layer->bias = malloc(*output_neurons * sizeof(float));
    for (int i=0; i<*output_neurons; i++) {
        layer->bias[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    // Initialize bias gradients
    layer->bias_gradients = malloc(*output_neurons * sizeof(float));
    for (int i=0; i<*output_neurons; i++) {
        layer->bias_gradients[i] = 0.0;
    }

    // Initalize backprop cache
    layer->backprop_cache = malloc(sizeof(BackpropCache));
    layer->backprop_cache->x_array = malloc(*input_neurons * sizeof(float));
    layer->backprop_cache->output = malloc(*output_neurons * sizeof(float));
    layer->backprop_cache->logits = malloc(*output_neurons * sizeof(float));

    for (int i=0; i<*input_neurons; i++) {
        layer->backprop_cache->x_array[i] = 0.0;
    }
    for (int i=0; i<*output_neurons; i++) {
        layer->backprop_cache->output[i] = 0.0;
        layer->backprop_cache->logits[i] = 0.0;
    }

    // Initialize gradients
    layer->gradients = malloc(*output_neurons * sizeof(float*));
    for (int i=0; i<*output_neurons; i++) {
        layer->gradients[i] = malloc(*input_neurons * sizeof(float));
        for (int j=0; j<*input_neurons; j++) {
            layer->gradients[i][j] = 0.0;
        }
    }

    layer->activation_function = activation_function;
    return layer;
}
