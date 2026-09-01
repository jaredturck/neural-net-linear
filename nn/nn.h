#ifndef NN_H
#define NN_H

#include "activation.h"

typedef struct Model Model;

Model* create_model(int input_size);
void add_linear(Model* model, int output_size, ActivationType activation_type);
void train(Model* model, float* train_x, float* train_y, int dataset_size, int epochs, float learning_rate);
void forward(Model* model, float* input, float* output);
void free_model(Model* model);

#endif
