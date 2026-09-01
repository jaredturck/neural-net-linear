#ifndef DATALOADER_H
#define DATALOADER_H

#include "datasets.h"
#include "samplers.h"

typedef struct DataLoader DataLoader;

DataLoader* create_dataloader(Dataset* dataset, Sampler* sampler, int batch_size, int drop_last);
void dataloader_reset(DataLoader* loader);
int dataloader_next(DataLoader* loader);
float* dataloader_x(DataLoader* loader);
float* dataloader_y(DataLoader* loader);
int dataloader_batch_size(DataLoader* loader);
int dataloader_input_size(DataLoader* loader);
int dataloader_output_size(DataLoader* loader);
void free_dataloader(DataLoader* loader);

#endif
