#ifndef TRANSFORMER_INTERNAL_H
#define TRANSFORMER_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "transformer.h"

#define GPT_RMS_EPSILON 1.0e-5f
#define GPT_ROPE_THETA 10000.0f

typedef struct {
    float* data;
    float* grad;
    float* first_moment;
    float* second_moment;
    size_t count;
    int weight_decay;
} Parameter;

typedef struct {
    Parameter* rms_attention;
    Parameter* qkv;
    Parameter* attention_output;
    Parameter* rms_feed_forward;
    Parameter* feed_forward_gate;
    Parameter* feed_forward_value;
    Parameter* feed_forward_output;
} GPTBlock;

struct GPTModel {
    int vocab_size;
    int context_length;
    int embedding_dim;
    int heads;
    int head_dim;
    int layers;
    int hidden_dim;
    uint64_t optimizer_step;

    Parameter* token_embedding;
    GPTBlock* blocks;
    Parameter* final_rms;

    Parameter** parameters;
    int parameter_count;
    int parameter_capacity;
};

typedef struct {
    float* norm_attention;
    float* qkv;
    float* attention_probabilities;
    float* attention_context;
    float* after_attention;
    float* norm_feed_forward;
    float* feed_forward_gate;
    float* feed_forward_value;
    float* feed_forward_product;
    float* inverse_rms_attention;
    float* inverse_rms_feed_forward;
} BlockCache;

typedef struct {
    int batch;
    int sequence;
    size_t rows;
    float* states;
    BlockCache* blocks;
    float* final_norm;
    float* final_inverse_rms;

    float* gradient_a;
    float* gradient_b;
    float* gradient_c;
    float* gradient_qkv;
    float* gradient_hidden_a;
    float* gradient_hidden_b;
    float* gradient_hidden_c;
    float* attention_scratch;
    float* logits;
} GPTWorkspace;

typedef struct {
    float score;
    int token;
} TokenScore;

typedef struct {
    uint32_t state;
} RandomState;

uint32_t random_next(RandomState* random);
int random_bounded(RandomState* random, int limit);
float random_uniform(RandomState* random);
void free_parameter(Parameter* parameter);
Parameter* add_parameter(GPTModel* model, size_t count, int weight_decay);
void initialize_matrix(Parameter* parameter, int input_size, int output_size, RandomState* random);
void initialize_norm(Parameter* parameter);
void zero_parameter_gradients(GPTModel* model);
float gradient_norm(GPTModel* model);
void scale_gradients(GPTModel* model, float scale);
void adamw_step(GPTModel* model, float learning_rate, float weight_decay, float beta1, float beta2, float epsilon);

void rms_norm_forward(const float* input, const float* weight, float* output, float* inverse_rms, size_t rows, int width);
void rms_norm_backward(const float* input, const float* weight, const float* inverse_rms, const float* output_gradient, float* input_gradient, float* weight_gradient, size_t rows, int width);
float silu(float value);
float silu_derivative(float value);
void rope_apply(float* qkv, int batch, int sequence, int embedding_dim, int heads, int inverse);
void attention_forward(const float* qkv, float* probabilities, float* context, int batch, int sequence, int embedding_dim, int heads);
void attention_backward(const float* qkv, const float* probabilities, const float* context_gradient, float* qkv_gradient, float* scratch, int batch, int sequence, int embedding_dim, int heads);

GPTWorkspace* create_workspace(GPTModel* model, int batch, int sequence);
void free_workspace(GPTModel* model, GPTWorkspace* workspace);
int forward_transformer(GPTModel* model, const int32_t* tokens, GPTWorkspace* workspace);
float train_batch(GPTModel* model, const int32_t* input_tokens, const int32_t* target_tokens, GPTWorkspace* workspace);

#endif
