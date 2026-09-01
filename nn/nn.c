#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nn.h"
#include "layers.h"
#include "backprop.h"
#include "loss.h"
#include "optimizers.h"

struct Model {
    Layer** layers;
    int layer_count;
    int layer_capacity;
    int input_size;
    int output_size;
};

static float* forward_internal(Model* model, float* input) {
    float* output = input;

    for (int i=0; i<model->layer_count; i++) {
        output = Linear(model->layers[i], output);
    }

    return output;
}

static void backward(Model* model, float* y_array) {
    Layer* output_layer = model->layers[model->layer_count - 1];
    compute_softmax_gradients(output_layer, y_array);

    for (int i=model->layer_count - 2; i>=0; i--) {
        Layer* layer = model->layers[i];
        Layer* next_layer = model->layers[i + 1];
        compute_layer_gradients(layer, next_layer->layer_deltas);
    }
}

static void optimizer_step(Model* model, float learning_rate) {
    for (int i=0; i<model->layer_count; i++) {
        SGD_layer(model->layers[i], learning_rate);
    }
}

Model* create_model(void) {
    srand((unsigned) time(NULL));

    Model* model = malloc(sizeof(Model));
    model->layer_capacity = 4;
    model->layers = malloc(model->layer_capacity * sizeof(Layer*));
    model->layer_count = 0;
    model->input_size = 0;
    model->output_size = 0;

    return model;
}

int add_linear(Model* model, int input_size, int output_size, ActivationType activation_type) {
    if (input_size <= 0 || output_size <= 0) {
        return 0;
    }

    if (model->layer_count > 0 && input_size != model->output_size) {
        return 0;
    }

    if (model->layer_count == model->layer_capacity) {
        model->layer_capacity *= 2;
        model->layers = realloc(model->layers, model->layer_capacity * sizeof(Layer*));
    }

    ActivationFunction* activation_function = get_activation_function(activation_type);
    Layer* layer = create_layer(input_size, output_size, activation_function, activation_type);

    if (model->layer_count == 0) {
        model->input_size = input_size;
    }

    model->layers[model->layer_count] = layer;
    model->layer_count++;
    model->output_size = output_size;

    return 1;
}

void train(Model* model, float* train_x, float* train_y, int dataset_size, int epochs, float learning_rate) {
    for (int epoch=0; epoch<epochs; epoch++) {
        float avg_loss = 0.0;

        for (int i=0; i<dataset_size; i++) {
            float* x = train_x + i * model->input_size;
            float* y = train_y + i * model->output_size;
            float* output = forward_internal(model, x);

            avg_loss += categorical_cross_entropy(y, output, model->output_size);
            backward(model, y);
            optimizer_step(model, learning_rate);
        }

        avg_loss /= dataset_size;
        if (epoch % 100 == 0) {
            printf("Epoch %d, loss %f\n", epoch + 1, avg_loss);
        }
        if (avg_loss <= 0.05) {
            printf("Training complete at epoch %d, loss %f\n", epoch + 1, avg_loss);
            return;
        }
    }
}

void train_dataset(Model* model, Dataset* dataset, int epochs, float learning_rate) {
    train(
        model,
        dataset_train_x(dataset),
        dataset_train_y(dataset),
        dataset_size(dataset),
        epochs,
        learning_rate
    );
}

void forward(Model* model, float* input, float* output) {
    float* model_output = forward_internal(model, input);
    memcpy(output, model_output, model->output_size * sizeof(float));
}

void free_model(Model* model) {
    for (int i=0; i<model->layer_count; i++) {
        free_layer(model->layers[i]);
    }

    free(model->layers);
    free(model);
}
