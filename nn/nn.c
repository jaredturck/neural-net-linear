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
    for (int i = 0; i < model->layer_count; i++) {
        output = Linear(model->layers[i], output);
    }
    return output;
}

static void backward(Model* model, float* y_array) {
    Layer* output_layer = model->layers[model->layer_count - 1];
    compute_softmax_gradients(output_layer, y_array);

    for (int i = model->layer_count - 2; i >= 0; i--) {
        Layer* layer = model->layers[i];
        Layer* next_layer = model->layers[i + 1];
        compute_layer_gradients(layer, next_layer->layer_deltas);
    }
}

static void zero_gradients(Model* model) {
    for (int i = 0; i < model->layer_count; i++) {
        zero_layer_gradients(model->layers[i]);
    }
}

static void average_gradients(Model* model, int batch_size) {
    float scale = 1.0f / (float)batch_size;
    for (int i = 0; i < model->layer_count; i++) {
        scale_layer_gradients(model->layers[i], scale);
    }
}

static void optimizer_step(Model* model, float learning_rate) {
    for (int i = 0; i < model->layer_count; i++) {
        SGD_layer(model->layers[i], learning_rate);
    }
}

static float train_batch(
    Model* model,
    float* batch_x,
    float* batch_y,
    int batch_size,
    float learning_rate
) {
    zero_gradients(model);
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; i++) {
        float* x = batch_x + (size_t)i * (size_t)model->input_size;
        float* y = batch_y + (size_t)i * (size_t)model->output_size;
        float* output = forward_internal(model, x);
        total_loss += categorical_cross_entropy(y, output, model->output_size);
        backward(model, y);
    }

    average_gradients(model, batch_size);
    optimizer_step(model, learning_rate);
    return total_loss;
}

Model* create_model(void) {
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned)time(NULL));
        seeded = 1;
    }

    Model* model = malloc(sizeof(Model));
    if (model == NULL) {
        return NULL;
    }
    model->layer_capacity = 4;
    model->layers = malloc((size_t)model->layer_capacity * sizeof(Layer*));
    if (model->layers == NULL) {
        free(model);
        return NULL;
    }
    model->layer_count = 0;
    model->input_size = 0;
    model->output_size = 0;
    return model;
}

int add_linear(Model* model, int input_size, int output_size, ActivationType activation_type) {
    if (model == NULL || input_size <= 0 || output_size <= 0) {
        return 0;
    }
    if (model->layer_count > 0 && input_size != model->output_size) {
        return 0;
    }

    if (model->layer_count == model->layer_capacity) {
        int new_capacity = model->layer_capacity * 2;
        Layer** layers = realloc(model->layers, (size_t)new_capacity * sizeof(Layer*));
        if (layers == NULL) {
            return 0;
        }
        model->layers = layers;
        model->layer_capacity = new_capacity;
    }

    ActivationFunction* activation_function = get_activation_function(activation_type);
    Layer* layer = create_layer(input_size, output_size, activation_function, activation_type);
    if (layer == NULL) {
        return 0;
    }

    if (model->layer_count == 0) {
        model->input_size = input_size;
    }
    model->layers[model->layer_count++] = layer;
    model->output_size = output_size;
    return 1;
}

void train(
    Model* model,
    float* train_x,
    float* train_y,
    int dataset_size,
    int epochs,
    float learning_rate
) {
    if (model == NULL || model->layer_count == 0 || train_x == NULL || train_y == NULL ||
        dataset_size <= 0 || epochs <= 0 || learning_rate <= 0.0f) {
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        for (int i = 0; i < dataset_size; i++) {
            total_loss += train_batch(
                model,
                train_x + (size_t)i * (size_t)model->input_size,
                train_y + (size_t)i * (size_t)model->output_size,
                1,
                learning_rate
            );
        }

        float avg_loss = total_loss / (float)dataset_size;
        if (epoch % 100 == 0) {
            printf("Epoch %d, loss %f\n", epoch + 1, avg_loss);
        }
        if (avg_loss <= 0.05f) {
            printf("Training complete at epoch %d, loss %f\n", epoch + 1, avg_loss);
            return;
        }
    }
}

int train_loader(Model* model, DataLoader* loader, int epochs, float learning_rate) {
    if (model == NULL || loader == NULL || model->layer_count == 0 ||
        epochs <= 0 || learning_rate <= 0.0f) {
        return 0;
    }
    if (dataloader_input_size(loader) != model->input_size ||
        dataloader_output_size(loader) != model->output_size) {
        return 0;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        dataloader_reset(loader);
        float total_loss = 0.0f;
        int sample_count = 0;
        int batch_size;

        while ((batch_size = dataloader_next(loader)) > 0) {
            total_loss += train_batch(
                model,
                dataloader_x(loader),
                dataloader_y(loader),
                batch_size,
                learning_rate
            );
            sample_count += batch_size;
        }

        if (sample_count == 0) {
            return 0;
        }

        float avg_loss = total_loss / (float)sample_count;
        if (epoch % 100 == 0) {
            printf("Epoch %d, loss %f\n", epoch + 1, avg_loss);
        }
        if (avg_loss <= 0.05f) {
            printf("Training complete at epoch %d, loss %f\n", epoch + 1, avg_loss);
            return 1;
        }
    }
    return 1;
}

void forward(Model* model, float* input, float* output) {
    if (model == NULL || model->layer_count == 0 || input == NULL || output == NULL) {
        return;
    }
    float* model_output = forward_internal(model, input);
    memcpy(output, model_output, (size_t)model->output_size * sizeof(float));
}

void free_model(Model* model) {
    if (model == NULL) {
        return;
    }
    for (int i = 0; i < model->layer_count; i++) {
        free_layer(model->layers[i]);
    }
    free(model->layers);
    free(model);
}
