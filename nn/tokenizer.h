#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

typedef struct BPETokenizer BPETokenizer;

BPETokenizer* bpe_train_file(const char* path, int target_vocab_size);
BPETokenizer* bpe_load(const char* path);
int bpe_save(BPETokenizer* tokenizer, const char* path);
int bpe_vocab_size(BPETokenizer* tokenizer);
int bpe_merge_count(BPETokenizer* tokenizer);
uint64_t bpe_fingerprint(BPETokenizer* tokenizer);

int bpe_encode(
    BPETokenizer* tokenizer,
    const unsigned char* input,
    int input_length,
    int32_t* output,
    int output_capacity
);

int bpe_encode_file(BPETokenizer* tokenizer, const char* path, int32_t** output_tokens);

int bpe_decode(
    BPETokenizer* tokenizer,
    const int32_t* tokens,
    int token_count,
    unsigned char* output,
    int output_capacity
);

void free_bpe_tokenizer(BPETokenizer* tokenizer);

#endif
