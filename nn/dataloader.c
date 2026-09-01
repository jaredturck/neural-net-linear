#include <stdlib.h>
#include "dataloader.h"

struct DataLoader {
    Dataset* dataset;
    Sampler* sampler;
    int batch_capacity;
    int current_batch_size;
    int input_size;
    int output_size;
    int drop_last;
    float* x;
    float* y;
};

DataLoader* create_dataloader(Dataset* dataset, Sampler* sampler, int batch_size, int drop_last) {
    if (dataset == NULL || sampler == NULL || batch_size <= 0) {
        return NULL;
    }

    DataLoader* loader = calloc(1, sizeof(DataLoader));
    if (loader == NULL) {
        return NULL;
    }
    loader->dataset = dataset;
    loader->sampler = sampler;
    loader->batch_capacity = batch_size;
    loader->drop_last = drop_last != 0;

    if (sampler_kind(sampler) == SAMPLER_ROWS) {
        if (dataset_kind(dataset) != DATASET_TABLE) {
            free(loader);
            return NULL;
        }
        loader->input_size = dataset_input_size(dataset);
        loader->output_size = dataset_output_size(dataset);
    } else {
        if (dataset_kind(dataset) != DATASET_TEXT) {
            free(loader);
            return NULL;
        }
        loader->input_size = sampler_sequence_length(sampler);
        loader->output_size = sampler_sequence_length(sampler);
    }

    loader->x = malloc((size_t)batch_size * (size_t)loader->input_size * sizeof(float));
    loader->y = malloc((size_t)batch_size * (size_t)loader->output_size * sizeof(float));
    if (loader->x == NULL || loader->y == NULL) {
        free(loader->x);
        free(loader->y);
        free(loader);
        return NULL;
    }

    return loader;
}

void dataloader_reset(DataLoader* loader) {
    if (loader == NULL) {
        return;
    }
    loader->current_batch_size = 0;
    sampler_reset(loader->sampler);
}

int dataloader_next(DataLoader* loader) {
    if (loader == NULL) {
        return 0;
    }

    int count = 0;
    int sample_index;
    while (count < loader->batch_capacity && sampler_next(loader->sampler, &sample_index)) {
        float* x = loader->x + (size_t)count * (size_t)loader->input_size;
        float* y = loader->y + (size_t)count * (size_t)loader->output_size;
        int copied;
        if (sampler_kind(loader->sampler) == SAMPLER_ROWS) {
            copied = dataset_copy_row(loader->dataset, sample_index, x, y);
        } else {
            copied = dataset_copy_token_window(loader->dataset, sample_index, loader->input_size, x, y);
        }
        if (!copied) {
            loader->current_batch_size = 0;
            return 0;
        }
        count++;
    }

    if (count == 0 || (loader->drop_last && count < loader->batch_capacity)) {
        loader->current_batch_size = 0;
        return 0;
    }

    loader->current_batch_size = count;
    return count;
}

float* dataloader_x(DataLoader* loader) {
    return loader == NULL ? NULL : loader->x;
}

float* dataloader_y(DataLoader* loader) {
    return loader == NULL ? NULL : loader->y;
}

int dataloader_batch_size(DataLoader* loader) {
    return loader == NULL ? 0 : loader->current_batch_size;
}

int dataloader_input_size(DataLoader* loader) {
    return loader == NULL ? 0 : loader->input_size;
}

int dataloader_output_size(DataLoader* loader) {
    return loader == NULL ? 0 : loader->output_size;
}

void free_dataloader(DataLoader* loader) {
    if (loader == NULL) {
        return;
    }
    free(loader->x);
    free(loader->y);
    free(loader);
}
