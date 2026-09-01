#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datasets.h"

#define ANIMAL_INPUT_SIZE 18
#define ANIMAL_OUTPUT_SIZE 12
#define LINE_SIZE 128

static const char* labels[] = {
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
};

static int label_index(const char* label) {
    for (int i=0; i<ANIMAL_OUTPUT_SIZE; i++) {
        if (strcmp(label, labels[i]) == 0) {
            return i;
        }
    }
    return 0;
}

void dataset_tokenize(const char* text, float* output, int input_size) {
    for (int i=0; i<input_size; i++) {
        output[i] = 0.0;
    }

    for (int i=0; text[i] != '\0' && i<input_size; i++) {
        if (text[i] == ' ') {
            output[i] = 26.0;
        } else {
            output[i] = (float)(text[i] - 'a');
        }
    }
}

Dataset* load_animal_dataset(const char* path) {
    FILE* file = fopen(path, "r");
    char line[LINE_SIZE];
    int size = 0;

    while (fgets(line, LINE_SIZE, file) != NULL) {
        size++;
    }
    rewind(file);

    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->size = size;
    dataset->input_size = ANIMAL_INPUT_SIZE;
    dataset->output_size = ANIMAL_OUTPUT_SIZE;
    dataset->train_x = calloc(size * ANIMAL_INPUT_SIZE, sizeof(float));
    dataset->train_y = calloc(size * ANIMAL_OUTPUT_SIZE, sizeof(float));

    int row = 0;
    while (fgets(line, LINE_SIZE, file) != NULL) {
        char* comma = strchr(line, ',');
        *comma = '\0';

        char* animal = line;
        char* family = comma + 1;
        family[strcspn(family, "\r\n")] = '\0';

        float* x = dataset->train_x + row * ANIMAL_INPUT_SIZE;
        float* y = dataset->train_y + row * ANIMAL_OUTPUT_SIZE;

        dataset_tokenize(animal, x, ANIMAL_INPUT_SIZE);
        y[label_index(family)] = 1.0;
        row++;
    }

    fclose(file);
    return dataset;
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

int dataset_argmax(float* values, int size) {
    int max_index = 0;
    float max_value = values[0];

    for (int i=1; i<size; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
            max_index = i;
        }
    }
    return max_index;
}

const char* dataset_label(int index) {
    return labels[index];
}

void free_dataset(Dataset* dataset) {
    free(dataset->train_x);
    free(dataset->train_y);
    free(dataset);
}
