#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datasets.h"

typedef struct {
    char** values;
    int count;
    int capacity;
} LabelMap;

typedef struct {
    int column;
    DataType type;
    int width;
    int offset;
    LabelMap labels;
} FieldSpec;

struct Dataset {
    DatasetKind kind;
    float* x;
    float* y;
    int size;
    int input_size;
    int output_size;

    FieldSpec* x_fields;
    int x_field_count;
    FieldSpec* y_fields;
    int y_field_count;

    float* tokens;
    int token_count;
    int vocab_size;
    TokenizerType tokenizer;
};

static char* copy_string(const char* value) {
    size_t length = strlen(value);
    char* copy = malloc(length + 1);
    if (copy != NULL) {
        memcpy(copy, value, length + 1);
    }
    return copy;
}

static void clear_label_map(LabelMap* map) {
    for (int i=0; i<map->count; i++) {
        free(map->values[i]);
    }
    free(map->values);
    map->values = NULL;
    map->count = 0;
    map->capacity = 0;
}

static void clear_fields(FieldSpec* fields, int count) {
    if (fields == NULL) {
        return;
    }
    for (int i=0; i<count; i++) {
        clear_label_map(&fields[i].labels);
    }
    free(fields);
}

static void clear_dataset(Dataset* dataset) {
    free(dataset->x);
    free(dataset->y);
    free(dataset->tokens);
    clear_fields(dataset->x_fields, dataset->x_field_count);
    clear_fields(dataset->y_fields, dataset->y_field_count);
    memset(dataset, 0, sizeof(Dataset));
}

static int label_index(LabelMap* map, const char* value, int add) {
    for (int i=0; i<map->count; i++) {
        if (strcmp(map->values[i], value) == 0) {
            return i;
        }
    }

    if (!add) {
        return -1;
    }

    if (map->count == map->capacity) {
        int new_capacity = map->capacity == 0 ? 8 : map->capacity * 2;
        char** values = realloc(map->values, (size_t)new_capacity * sizeof(char*));
        if (values == NULL) {
            return -1;
        }
        map->values = values;
        map->capacity = new_capacity;
    }

    char* copy = copy_string(value);
    if (copy == NULL) {
        return -1;
    }
    map->values[map->count] = copy;
    return map->count++;
}

static char* read_line(FILE* file) {
    size_t capacity = 128;
    size_t length = 0;
    char* line = malloc(capacity);
    if (line == NULL) {
        return NULL;
    }

    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (length + 1 >= capacity) {
            capacity *= 2;
            char* resized = realloc(line, capacity);
            if (resized == NULL) {
                free(line);
                return NULL;
            }
            line = resized;
        }
        if (ch == '\n') {
            break;
        }
        if (ch != '\r') {
            line[length++] = (char)ch;
        }
    }

    if (ch == EOF && length == 0) {
        free(line);
        return NULL;
    }

    line[length] = '\0';
    return line;
}

static int split_delimited(char* line, char delimiter, char*** output_fields) {
    int capacity = 8;
    int count = 0;
    char** fields = malloc((size_t)capacity * sizeof(char*));
    if (fields == NULL) {
        return -1;
    }

    char* read = line;
    char* write = line;
    char* field_start = write;
    int in_quotes = 0;

    for (;;) {
        char ch = *read++;
        if (ch == '"') {
            if (in_quotes && *read == '"') {
                *write++ = '"';
                read++;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }

        if ((ch == delimiter && !in_quotes) || ch == '\0') {
            *write++ = '\0';
            if (count == capacity) {
                capacity *= 2;
                char** resized = realloc(fields, (size_t)capacity * sizeof(char*));
                if (resized == NULL) {
                    free(fields);
                    return -1;
                }
                fields = resized;
            }
            fields[count++] = field_start;
            field_start = write;
            if (ch == '\0') {
                break;
            }
            continue;
        }

        *write++ = ch;
    }

    if (in_quotes) {
        free(fields);
        return -1;
    }

    *output_fields = fields;
    return count;
}

static FieldSpec* create_fields(
    const int* columns,
    const DataType* types,
    const int* widths,
    int count
) {
    if (count <= 0) {
        return NULL;
    }

    FieldSpec* fields = calloc((size_t)count, sizeof(FieldSpec));
    if (fields == NULL) {
        return NULL;
    }

    for (int i=0; i<count; i++) {
        fields[i].column = columns[i];
        fields[i].type = types[i];
        fields[i].width = widths == NULL ? 0 : widths[i];
        if (fields[i].type != DATA_TEXT) {
            fields[i].width = 1;
        }
        if (fields[i].column < 0 || fields[i].width < 0) {
            clear_fields(fields, count);
            return NULL;
        }
    }
    return fields;
}

static int max_selected_column(Dataset* dataset) {
    int maximum = -1;
    for (int i=0; i<dataset->x_field_count; i++) {
        if (dataset->x_fields[i].column > maximum) {
            maximum = dataset->x_fields[i].column;
        }
    }
    for (int i=0; i<dataset->y_field_count; i++) {
        if (dataset->y_fields[i].column > maximum) {
            maximum = dataset->y_fields[i].column;
        }
    }
    return maximum;
}

static int scan_fields(FieldSpec* fields, int count, char** columns, int column_count) {
    for (int i=0; i<count; i++) {
        FieldSpec* field = &fields[i];
        if (field->column >= column_count) {
            return 0;
        }
        if (field->type == DATA_LABEL && label_index(&field->labels, columns[field->column], 1) < 0) {
            return 0;
        }
        if (field->type == DATA_TEXT && field->width == 0) {
            int length = (int)strlen(columns[field->column]);
            if (length > field->offset) {
                field->offset = length;
            }
        }
    }
    return 1;
}

static int calculate_layout(FieldSpec* fields, int count) {
    int offset = 0;
    for (int i=0; i<count; i++) {
        int inferred_text_width = fields[i].type == DATA_TEXT && fields[i].width == 0 ? fields[i].offset : fields[i].width;
        fields[i].offset = offset;
        if (fields[i].type == DATA_LABEL) {
            fields[i].width = fields[i].labels.count;
        } else if (fields[i].type == DATA_TEXT) {
            fields[i].width = inferred_text_width;
        }
        if (fields[i].width <= 0) {
            return -1;
        }
        offset += fields[i].width;
    }
    return offset;
}

static int encode_field(FieldSpec* field, const char* value, float* output) {
    char* end = NULL;
    switch (field->type) {
        case DATA_FLOAT: {
            float parsed = strtof(value, &end);
            if (end == value || *end != '\0') {
                return 0;
            }
            output[0] = parsed;
            return 1;
        }
        case DATA_INT: {
            long parsed = strtol(value, &end, 10);
            if (end == value || *end != '\0') {
                return 0;
            }
            output[0] = (float)parsed;
            return 1;
        }
        case DATA_TEXT:
            for (int i=0; i<field->width; i++) {
                output[i] = 0.0f;
            }
            for (int i=0; value[i] != '\0' && i<field->width; i++) {
                output[i] = (float)((unsigned char)value[i] + 1u) / 256.0f;
            }
            return 1;
        case DATA_LABEL: {
            for (int i=0; i<field->width; i++) {
                output[i] = 0.0f;
            }
            int index = label_index(&field->labels, value, 0);
            if (index < 0) {
                return 0;
            }
            output[index] = 1.0f;
            return 1;
        }
    }
    return 0;
}

static int encode_fields(FieldSpec* fields, int count, char** columns, int column_count, float* output) {
    for (int i=0; i<count; i++) {
        FieldSpec* field = &fields[i];
        if (field->column >= column_count || !encode_field(field, columns[field->column], output + field->offset)) {
            return 0;
        }
    }
    return 1;
}

Dataset* create_dataset(void) {
    return calloc(1, sizeof(Dataset));
}

int load_csv_dataset(
    Dataset* dataset,
    const char* path,
    char delimiter,
    const int* x_columns,
    const DataType* x_types,
    const int* x_widths,
    int x_count,
    const int* y_columns,
    const DataType* y_types,
    const int* y_widths,
    int y_count,
    int has_header
) {
    if (dataset == NULL || path == NULL || delimiter == '\0' || x_count <= 0 || y_count <= 0) {
        return 0;
    }

    FILE* file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }

    clear_dataset(dataset);
    dataset->x_field_count = x_count;
    dataset->y_field_count = y_count;
    dataset->x_fields = create_fields(x_columns, x_types, x_widths, x_count);
    dataset->y_fields = create_fields(y_columns, y_types, y_widths, y_count);
    if (dataset->x_fields == NULL || dataset->y_fields == NULL) {
        fclose(file);
        clear_dataset(dataset);
        return 0;
    }

    int required_columns = max_selected_column(dataset) + 1;
    int row_count = 0;
    int row_number = 0;
    char* line;
    while ((line = read_line(file)) != NULL) {
        row_number++;
        if (has_header && row_number == 1) {
            free(line);
            continue;
        }
        if (line[0] == '\0') {
            free(line);
            continue;
        }

        char** columns = NULL;
        int column_count = split_delimited(line, delimiter, &columns);
        if (column_count < required_columns ||
            !scan_fields(dataset->x_fields, x_count, columns, column_count) ||
            !scan_fields(dataset->y_fields, y_count, columns, column_count)) {
            free(columns);
            free(line);
            fclose(file);
            clear_dataset(dataset);
            return 0;
        }
        row_count++;
        free(columns);
        free(line);
    }

    if (row_count == 0) {
        fclose(file);
        clear_dataset(dataset);
        return 0;
    }

    dataset->input_size = calculate_layout(dataset->x_fields, x_count);
    dataset->output_size = calculate_layout(dataset->y_fields, y_count);
    if (dataset->input_size <= 0 || dataset->output_size <= 0) {
        fclose(file);
        clear_dataset(dataset);
        return 0;
    }
    dataset->size = row_count;
    dataset->kind = DATASET_TABLE;
    dataset->x = calloc((size_t)row_count * (size_t)dataset->input_size, sizeof(float));
    dataset->y = calloc((size_t)row_count * (size_t)dataset->output_size, sizeof(float));
    if (dataset->x == NULL || dataset->y == NULL) {
        fclose(file);
        clear_dataset(dataset);
        return 0;
    }

    rewind(file);
    row_number = 0;
    int row = 0;
    while ((line = read_line(file)) != NULL) {
        row_number++;
        if ((has_header && row_number == 1) || line[0] == '\0') {
            free(line);
            continue;
        }

        char** columns = NULL;
        int column_count = split_delimited(line, delimiter, &columns);
        float* x = dataset->x + (size_t)row * (size_t)dataset->input_size;
        float* y = dataset->y + (size_t)row * (size_t)dataset->output_size;
        int valid = column_count >= required_columns &&
            encode_fields(dataset->x_fields, x_count, columns, column_count, x) &&
            encode_fields(dataset->y_fields, y_count, columns, column_count, y);
        free(columns);
        free(line);
        if (!valid) {
            fclose(file);
            clear_dataset(dataset);
            return 0;
        }
        row++;
    }

    fclose(file);
    return 1;
}

int load_text_dataset(Dataset* dataset, const char* path, TokenizerType tokenizer) {
    if (dataset == NULL || path == NULL || tokenizer != TOKENIZER_BYTE) {
        return 0;
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    clear_dataset(dataset);
    size_t capacity = 4096;
    size_t count = 0;
    float* tokens = malloc(capacity * sizeof(float));
    if (tokens == NULL) {
        fclose(file);
        return 0;
    }

    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (count == capacity) {
            capacity *= 2;
            float* resized = realloc(tokens, capacity * sizeof(float));
            if (resized == NULL) {
                free(tokens);
                fclose(file);
                return 0;
            }
            tokens = resized;
        }
        tokens[count++] = (float)((unsigned char)ch + 1u);
    }
    fclose(file);

    if (count < 2) {
        free(tokens);
        return 0;
    }

    dataset->kind = DATASET_TEXT;
    dataset->tokens = tokens;
    dataset->token_count = (int)count;
    dataset->vocab_size = 257;
    dataset->tokenizer = tokenizer;
    return 1;
}

DatasetKind dataset_kind(Dataset* dataset) {
    return dataset == NULL ? DATASET_EMPTY : dataset->kind;
}

int dataset_size(Dataset* dataset) {
    return dataset == NULL ? 0 : dataset->size;
}

int dataset_input_size(Dataset* dataset) {
    return dataset == NULL ? 0 : dataset->input_size;
}

int dataset_output_size(Dataset* dataset) {
    return dataset == NULL ? 0 : dataset->output_size;
}

int dataset_token_count(Dataset* dataset) {
    return dataset == NULL ? 0 : dataset->token_count;
}

int dataset_vocab_size(Dataset* dataset) {
    return dataset == NULL ? 0 : dataset->vocab_size;
}

int dataset_copy_row(Dataset* dataset, int index, float* x, float* y) {
    if (dataset == NULL || dataset->kind != DATASET_TABLE || index < 0 || index >= dataset->size) {
        return 0;
    }
    memcpy(x, dataset->x + (size_t)index * (size_t)dataset->input_size,
        (size_t)dataset->input_size * sizeof(float));
    memcpy(y, dataset->y + (size_t)index * (size_t)dataset->output_size,
        (size_t)dataset->output_size * sizeof(float));
    return 1;
}

int dataset_copy_token_window(Dataset* dataset, int start, int sequence_length, float* x, float* y) {
    if (dataset == NULL || dataset->kind != DATASET_TEXT || sequence_length <= 0 || start < 0 ||
        start + sequence_length >= dataset->token_count) {
        return 0;
    }
    memcpy(x, dataset->tokens + start, (size_t)sequence_length * sizeof(float));
    memcpy(y, dataset->tokens + start + 1, (size_t)sequence_length * sizeof(float));
    return 1;
}

void dataset_encode_text(Dataset* dataset, const char* text, float* output) {
    if (dataset == NULL || dataset->kind != DATASET_TABLE || text == NULL || output == NULL) {
        return;
    }
    for (int i=0; i<dataset->input_size; i++) {
        output[i] = 0.0f;
    }
    for (int i=0; i<dataset->x_field_count; i++) {
        FieldSpec* field = &dataset->x_fields[i];
        if (field->type == DATA_TEXT) {
            encode_field(field, text, output + field->offset);
            return;
        }
    }
}

int dataset_argmax(Dataset* dataset, float* values) {
    if (dataset == NULL || values == NULL || dataset->output_size <= 0) {
        return -1;
    }
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
    if (dataset == NULL || dataset->kind != DATASET_TABLE) {
        return NULL;
    }
    for (int i=0; i<dataset->y_field_count; i++) {
        FieldSpec* field = &dataset->y_fields[i];
        if (field->type == DATA_LABEL) {
            if (index < field->offset || index >= field->offset + field->width) {
                return NULL;
            }
            return field->labels.values[index - field->offset];
        }
    }
    return NULL;
}

void free_dataset(Dataset* dataset) {
    if (dataset == NULL) {
        return;
    }
    clear_dataset(dataset);
    free(dataset);
}
