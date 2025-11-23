#include "activation.c";
#include "layers.c";

// Module - build 3 layer neural network
int main() {
    Layer* layer_1 = create_layer(18, 32, relu, F_RELU);
    Layer* layer_2 = create_layer(32, 32, relu, F_RELU);
    Layer* layer_3 = create_layer(32, 12, NULL, F_SOFTMAX);
}
