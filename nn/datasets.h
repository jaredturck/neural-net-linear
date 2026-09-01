#ifndef DATASETS_H
#define DATASETS_H

typedef enum {
    DATA_FLOAT,
    DATA_INT,
    DATA_TEXT,
    DATA_LABEL
} DataType;

typedef enum {
    DATASET_EMPTY,
    DATASET_TABLE,
    DATASET_TEXT
} DatasetKind;

typedef enum {
    TOKENIZER_BYTE
} TokenizerType;

typedef struct Dataset Dataset;

Dataset* create_dataset(void);
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
);
int load_text_dataset(Dataset* dataset, const char* path, TokenizerType tokenizer);
DatasetKind dataset_kind(Dataset* dataset);
int dataset_size(Dataset* dataset);
int dataset_input_size(Dataset* dataset);
int dataset_output_size(Dataset* dataset);
int dataset_token_count(Dataset* dataset);
int dataset_vocab_size(Dataset* dataset);
int dataset_copy_row(Dataset* dataset, int index, float* x, float* y);
int dataset_copy_token_window(Dataset* dataset, int start, int sequence_length, float* x, float* y);
void dataset_encode_text(Dataset* dataset, const char* text, float* output);
int dataset_argmax(Dataset* dataset, float* values);
const char* dataset_label(Dataset* dataset, int index);
void free_dataset(Dataset* dataset);

#endif
