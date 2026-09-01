#ifndef NN_H
#define NN_H

#include "activation.h"
#include "datasets.h"

typedef struct Model Model;

Model* create_model(void);
int add_linear(Model* model, int input_size, int output_size, ActivationType activation_type);
void train(Model* model, float* train_x, float* train_y, int dataset_size, int epochs, float learning_rate);
void train_dataset(Model* model, Dataset* dataset, int epochs, float learning_rate);
void forward(Model* model, float* input, float* output);
void free_model(Model* model);

#endif
