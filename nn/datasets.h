#ifndef DATASETS_H
#define DATASETS_H

typedef struct Dataset Dataset;

Dataset* load_animal_dataset(const char* path);
int dataset_size(Dataset* dataset);
int dataset_input_size(Dataset* dataset);
int dataset_output_size(Dataset* dataset);
float* dataset_train_x(Dataset* dataset);
float* dataset_train_y(Dataset* dataset);
void dataset_tokenize(const char* text, float* output, int input_size);
int dataset_argmax(float* values, int size);
const char* dataset_label(int index);
void free_dataset(Dataset* dataset);

#endif
