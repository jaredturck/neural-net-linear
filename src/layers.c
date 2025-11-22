
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
