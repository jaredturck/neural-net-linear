#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datasets.h"

#define ANIMAL_INPUT_SIZE 18
#define ANIMAL_OUTPUT_SIZE 12
#define LINE_SIZE 128

static const char* animal_labels[] = {
    "amphibian",
    "arachnid",
    "bird",
    "cnidarian",
    "crustacean",
    "echinoderm",
    "fish",
    "insect",
    "mammal",
    "marsupial",
    "mollusk",
    "reptile"
};

struct Dataset {
    float* train_x;
    float* train_y;
    int size;
    int input_size;
    int output_size;
    const char** labels;
};

static void clear_dataset(Dataset* dataset) {
    free(dataset->train_x);
    free(dataset->train_y);

    dataset->train_x = NULL;
    dataset->train_y = NULL;
    dataset->size = 0;
    dataset->input_size = 0;
    dataset->output_size = 0;
    dataset->labels = NULL;
}

static int label_index(Dataset* dataset, const char* label) {
    for (int i=0; i<dataset->output_size; i++) {
        if (strcmp(label, dataset->labels[i]) == 0) {
            return i;
        }
    }
    return -1;
}

Dataset* create_dataset(void) {
    return calloc(1, sizeof(Dataset));
}

void dataset_tokenize(Dataset* dataset, const char* text, float* output) {
    for (int i=0; i<dataset->input_size; i++) {
        output[i] = 0.0;
    }

    for (int i=0; text[i] != '\0' && i<dataset->input_size; i++) {
        if (text[i] == ' ') {
            output[i] = 26.0;
        } else {
            output[i] = (float)(text[i] - 'a');
        }
    }
}

int load_animal_dataset(Dataset* dataset, const char* path) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }

    clear_dataset(dataset);

    char line[LINE_SIZE];
    int size = 0;

    while (fgets(line, LINE_SIZE, file) != NULL) {
        size++;
    }
    rewind(file);

    dataset->size = size;
    dataset->input_size = ANIMAL_INPUT_SIZE;
    dataset->output_size = ANIMAL_OUTPUT_SIZE;
    dataset->labels = animal_labels;
    dataset->train_x = calloc(size * dataset->input_size, sizeof(float));
    dataset->train_y = calloc(size * dataset->output_size, sizeof(float));

    if (dataset->train_x == NULL || dataset->train_y == NULL) {
        fclose(file);
        clear_dataset(dataset);
        return 0;
    }

    int row = 0;
    while (fgets(line, LINE_SIZE, file) != NULL) {
        char* comma = strchr(line, ',');
        if (comma == NULL) {
            fclose(file);
            clear_dataset(dataset);
            return 0;
        }

        *comma = '\0';

        char* animal = line;
        char* family = comma + 1;
        family[strcspn(family, "\r\n")] = '\0';

        int family_index = label_index(dataset, family);
        if (family_index < 0) {
            fclose(file);
            clear_dataset(dataset);
            return 0;
        }

        float* x = dataset->train_x + row * dataset->input_size;
        float* y = dataset->train_y + row * dataset->output_size;

        dataset_tokenize(dataset, animal, x);
        y[family_index] = 1.0;
        row++;
    }

    fclose(file);
    return 1;
}

int dataset_size(Dataset* dataset) {
    return dataset->size;
}

int dataset_input_size(Dataset* dataset) {
    return dataset->input_size;
}

int dataset_output_size(Dataset* dataset) {
    return dataset->output_size;
}

float* dataset_train_x(Dataset* dataset) {
    return dataset->train_x;
}

float* dataset_train_y(Dataset* dataset) {
    return dataset->train_y;
}

int dataset_argmax(Dataset* dataset, float* values) {
    int max_index = 0;
    float max_value = values[0];

    for (int i=1; i<dataset->output_size; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
            max_index = i;
        }
    }
    return max_index;
}

const char* dataset_label(Dataset* dataset, int index) {
    if (index < 0 || index >= dataset->output_size) {
        return NULL;
    }
    return dataset->labels[index];
}

void free_dataset(Dataset* dataset) {
    clear_dataset(dataset);
    free(dataset);
}
