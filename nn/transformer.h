#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"

typedef struct GPTModel GPTModel;

typedef struct {
    int epochs;
    int batch_size;
    int steps_per_epoch;
    int log_every;
    int warmup_steps;
    float learning_rate;
    float weight_decay;
    float beta1;
    float beta2;
    float epsilon;
    float grad_clip;
    unsigned int seed;
} GPTTrainConfig;

GPTModel* create_gpt_model(
    int vocab_size,
    int context_length,
    int embedding_dim,
    int heads,
    int layers,
    int hidden_dim,
    unsigned int seed
);

int gpt_vocab_size(GPTModel* model);
int gpt_context_length(GPTModel* model);
int gpt_embedding_dim(GPTModel* model);
int gpt_head_count(GPTModel* model);
int gpt_layer_count(GPTModel* model);
int gpt_hidden_dim(GPTModel* model);

int gpt_train_file(
    GPTModel* model,
    BPETokenizer* tokenizer,
    const char* path,
    const GPTTrainConfig* config
);

unsigned char* gpt_generate(
    GPTModel* model,
    BPETokenizer* tokenizer,
    const unsigned char* prompt,
    int prompt_length,
    int max_new_tokens,
    float temperature,
    int top_k,
    unsigned int seed,
    int* output_length
);

int gpt_save(GPTModel* model, const char* path);
GPTModel* gpt_load(const char* path);
void gpt_free_bytes(unsigned char* bytes);
void free_gpt_model(GPTModel* model);

#endif
