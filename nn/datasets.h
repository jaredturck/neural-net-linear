#ifndef DATASETS_H
#define DATASETS_H

typedef struct Dataset Dataset;

Dataset* create_dataset(void);
int load_animal_dataset(Dataset* dataset, const char* path);
int dataset_size(Dataset* dataset);
int dataset_input_size(Dataset* dataset);
int dataset_output_size(Dataset* dataset);
float* dataset_train_x(Dataset* dataset);
float* dataset_train_y(Dataset* dataset);
void dataset_tokenize(Dataset* dataset, const char* text, float* output);
int dataset_argmax(Dataset* dataset, float* values);
const char* dataset_label(Dataset* dataset, int index);
void free_dataset(Dataset* dataset);

#endif
