#include <float.h>
#include <limits.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transformer_internal.h"

static int checked_size_multiply(size_t left, size_t right, size_t* output) {
    if (output == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return 0;
    }
    *output = left * right;
    return 1;
}

static Parameter* add_matrix_parameter(
    GPTModel* model,
    int rows,
    int columns,
    int weight_decay
) {
    size_t count = 0;
    if (rows <= 0 || columns <= 0 ||
        !checked_size_multiply((size_t)rows, (size_t)columns, &count)) {
        return NULL;
    }
    return add_parameter(model, count, weight_decay);
}

GPTModel* create_gpt_model(
    int vocab_size,
    int context_length,
    int embedding_dim,
    int heads,
    int layers,
    int hidden_dim,
    unsigned int seed
) {
    if (vocab_size < 2 || context_length <= 0 || embedding_dim <= 0 || heads <= 0 ||
        layers <= 0 || hidden_dim <= 0 || embedding_dim % heads != 0 ||
        (embedding_dim / heads) % 2 != 0 || embedding_dim > INT_MAX / 3 ||
        layers > (INT_MAX - 2) / 7) {
        return NULL;
    }

    GPTModel* model = calloc(1, sizeof(GPTModel));
    if (model == NULL) {
        return NULL;
    }
    model->vocab_size = vocab_size;
    model->context_length = context_length;
    model->embedding_dim = embedding_dim;
    model->heads = heads;
    model->head_dim = embedding_dim / heads;
    model->layers = layers;
    model->hidden_dim = hidden_dim;
    model->blocks = calloc((size_t)layers, sizeof(GPTBlock));
    if (model->blocks == NULL) {
        free_gpt_model(model);
        return NULL;
    }

    RandomState random = {seed == 0 ? 1u : seed};
    model->token_embedding = add_matrix_parameter(model, vocab_size, embedding_dim, 1);
    if (model->token_embedding == NULL) {
        free_gpt_model(model);
        return NULL;
    }
    initialize_matrix(model->token_embedding, embedding_dim, vocab_size, &random);

    for (int layer = 0; layer < layers; layer++) {
        GPTBlock* block = &model->blocks[layer];
        block->rms_attention = add_parameter(model, (size_t)embedding_dim, 0);
        block->qkv = add_matrix_parameter(model, 3 * embedding_dim, embedding_dim, 1);
        block->attention_output = add_matrix_parameter(model, embedding_dim, embedding_dim, 1);
        block->rms_feed_forward = add_parameter(model, (size_t)embedding_dim, 0);
        block->feed_forward_gate = add_matrix_parameter(model, hidden_dim, embedding_dim, 1);
        block->feed_forward_value = add_matrix_parameter(model, hidden_dim, embedding_dim, 1);
        block->feed_forward_output = add_matrix_parameter(model, embedding_dim, hidden_dim, 1);

        if (block->rms_attention == NULL || block->qkv == NULL ||
            block->attention_output == NULL || block->rms_feed_forward == NULL ||
            block->feed_forward_gate == NULL || block->feed_forward_value == NULL ||
            block->feed_forward_output == NULL) {
            free_gpt_model(model);
            return NULL;
        }

        initialize_norm(block->rms_attention);
        initialize_matrix(block->qkv, embedding_dim, 3 * embedding_dim, &random);
        initialize_matrix(block->attention_output, embedding_dim, embedding_dim, &random);
        initialize_norm(block->rms_feed_forward);
        initialize_matrix(block->feed_forward_gate, embedding_dim, hidden_dim, &random);
        initialize_matrix(block->feed_forward_value, embedding_dim, hidden_dim, &random);
        initialize_matrix(block->feed_forward_output, hidden_dim, embedding_dim, &random);
    }

    model->final_rms = add_parameter(model, (size_t)embedding_dim, 0);
    if (model->final_rms == NULL) {
        free_gpt_model(model);
        return NULL;
    }
    initialize_norm(model->final_rms);
    return model;
}

int gpt_vocab_size(GPTModel* model) { return model == NULL ? 0 : model->vocab_size; }
int gpt_context_length(GPTModel* model) { return model == NULL ? 0 : model->context_length; }
int gpt_embedding_dim(GPTModel* model) { return model == NULL ? 0 : model->embedding_dim; }
int gpt_head_count(GPTModel* model) { return model == NULL ? 0 : model->heads; }
int gpt_layer_count(GPTModel* model) { return model == NULL ? 0 : model->layers; }
int gpt_hidden_dim(GPTModel* model) { return model == NULL ? 0 : model->hidden_dim; }
uint64_t gpt_tokenizer_fingerprint(GPTModel* model) {
    return model == NULL ? 0 : model->tokenizer_fingerprint;
}

int gpt_bind_tokenizer(GPTModel* model, BPETokenizer* tokenizer) {
    if (model == NULL || tokenizer == NULL || bpe_vocab_size(tokenizer) != model->vocab_size) {
        return 0;
    }
    uint64_t fingerprint = bpe_fingerprint(tokenizer);
    if (fingerprint == 0) {
        return 0;
    }
    if (model->tokenizer_fingerprint == 0) {
        model->tokenizer_fingerprint = fingerprint;
        return 1;
    }
    return model->tokenizer_fingerprint == fingerprint;
}

static void shuffle_ints(int* values, int count, RandomState* random) {
    for (int i = count - 1; i > 0; i--) {
        int j = random_bounded(random, i + 1);
        int temporary = values[i];
        values[i] = values[j];
        values[j] = temporary;
    }
}

static void fill_batch(
    const int32_t* corpus,
    const int* starts,
    int batch_size,
    int sequence,
    int32_t* input,
    int32_t* target
) {
    for (int b = 0; b < batch_size; b++) {
        int start = starts[b];
        for (int t = 0; t < sequence; t++) {
            input[(size_t)b * (size_t)sequence + (size_t)t] = corpus[start + t];
            target[(size_t)b * (size_t)sequence + (size_t)t] = corpus[start + t + 1];
        }
    }
}

int gpt_train_file(
    GPTModel* model,
    BPETokenizer* tokenizer,
    const char* path,
    const GPTTrainConfig* config
) {
    if (model == NULL || tokenizer == NULL || path == NULL || config == NULL ||
        config->epochs <= 0 || config->batch_size <= 0 || config->steps_per_epoch < 0 ||
        config->log_every < 0 || config->warmup_steps < 0 ||
        !isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        !isfinite(config->weight_decay) || config->weight_decay < 0.0f ||
        !isfinite(config->beta1) || config->beta1 < 0.0f || config->beta1 >= 1.0f ||
        !isfinite(config->beta2) || config->beta2 < 0.0f || config->beta2 >= 1.0f ||
        !isfinite(config->epsilon) || config->epsilon <= 0.0f ||
        !isfinite(config->grad_clip) || config->grad_clip < 0.0f ||
        !parameters_are_finite(model) || !gpt_bind_tokenizer(model, tokenizer)) {
        return 0;
    }

    int32_t* corpus = NULL;
    int token_count = bpe_encode_file(tokenizer, path, &corpus);
    if (token_count <= model->context_length) {
        free(corpus);
        return 0;
    }

    int sequence = model->context_length;
    int block_count = (token_count - 1) / sequence;
    if (block_count <= 0) {
        free(corpus);
        return 0;
    }

    size_t batch_values = 0;
    size_t batch_bytes = 0;
    if (!checked_size_multiply(
            (size_t)config->batch_size, (size_t)sequence, &batch_values) ||
        !checked_size_multiply(batch_values, sizeof(int32_t), &batch_bytes)) {
        free(corpus);
        return 0;
    }

    size_t blocks_bytes = 0;
    size_t starts_bytes = 0;
    if (!checked_size_multiply((size_t)block_count, sizeof(int), &blocks_bytes) ||
        !checked_size_multiply((size_t)config->batch_size, sizeof(int), &starts_bytes)) {
        free(corpus);
        return 0;
    }

    int* blocks = malloc(blocks_bytes);
    int* batch_starts = malloc(starts_bytes);
    int32_t* input = malloc(batch_bytes);
    int32_t* target = malloc(batch_bytes);
    GPTWorkspace* full_workspace = create_workspace(model, config->batch_size, sequence);
    if (blocks == NULL || batch_starts == NULL || input == NULL || target == NULL ||
        full_workspace == NULL) {
        free(corpus);
        free(blocks);
        free(batch_starts);
        free(input);
        free(target);
        free_workspace(model, full_workspace);
        return 0;
    }

    for (int i = 0; i < block_count; i++) {
        blocks[i] = i * sequence;
    }

    RandomState random = {config->seed == 0 ? 1u : config->seed};
    uint64_t total_training_steps = 0;

    for (int epoch = 0; epoch < config->epochs; epoch++) {
        double epoch_loss = 0.0;
        uint64_t epoch_samples = 0;

        if (config->steps_per_epoch > 0) {
            for (int step = 0; step < config->steps_per_epoch; step++) {
                int max_start = token_count - sequence;
                for (int b = 0; b < config->batch_size; b++) {
                    batch_starts[b] = random_bounded(&random, max_start);
                }
                fill_batch(corpus, batch_starts, config->batch_size, sequence, input, target);
                float loss = train_batch(model, input, target, full_workspace);
                if (!isfinite(loss) || !gradients_are_finite(model)) {
                    goto training_failed;
                }

                float norm = config->grad_clip > 0.0f ? gradient_norm(model) : 0.0f;
                if (config->grad_clip > 0.0f && !isfinite(norm)) {
                    goto training_failed;
                }
                if (config->grad_clip > 0.0f && norm > config->grad_clip) {
                    scale_gradients(model, config->grad_clip / norm);
                }
                float learning_rate = config->learning_rate;
                if (config->warmup_steps > 0 &&
                    model->optimizer_step < (uint64_t)config->warmup_steps) {
                    learning_rate *= (float)(model->optimizer_step + 1) /
                        (float)config->warmup_steps;
                }
                if (!adamw_step(
                    model,
                    learning_rate,
                    config->weight_decay,
                    config->beta1,
                    config->beta2,
                    config->epsilon
                )) {
                    goto training_failed;
                }
                total_training_steps++;
                epoch_samples += (uint64_t)config->batch_size;
                epoch_loss += (double)loss * (double)config->batch_size;
                if (config->log_every > 0 && total_training_steps % (uint64_t)config->log_every == 0) {
                    printf("Step %" PRIu64 ", loss %f\n", total_training_steps, loss);
                    fflush(stdout);
                }
            }
        } else {
            shuffle_ints(blocks, block_count, &random);
            for (int offset = 0; offset < block_count; offset += config->batch_size) {
                int actual_batch = block_count - offset;
                if (actual_batch > config->batch_size) {
                    actual_batch = config->batch_size;
                }
                for (int b = 0; b < actual_batch; b++) {
                    batch_starts[b] = blocks[offset + b];
                }
                fill_batch(corpus, batch_starts, actual_batch, sequence, input, target);

                GPTWorkspace* workspace = full_workspace;
                if (actual_batch != config->batch_size) {
                    workspace = create_workspace(model, actual_batch, sequence);
                    if (workspace == NULL) {
                        goto training_failed;
                    }
                }

                float loss = train_batch(model, input, target, workspace);
                if (workspace != full_workspace) {
                    free_workspace(model, workspace);
                }
                if (!isfinite(loss) || !gradients_are_finite(model)) {
                    goto training_failed;
                }

                float norm = config->grad_clip > 0.0f ? gradient_norm(model) : 0.0f;
                if (config->grad_clip > 0.0f && !isfinite(norm)) {
                    goto training_failed;
                }
                if (config->grad_clip > 0.0f && norm > config->grad_clip) {
                    scale_gradients(model, config->grad_clip / norm);
                }
                float learning_rate = config->learning_rate;
                if (config->warmup_steps > 0 &&
                    model->optimizer_step < (uint64_t)config->warmup_steps) {
                    learning_rate *= (float)(model->optimizer_step + 1) /
                        (float)config->warmup_steps;
                }
                if (!adamw_step(
                    model,
                    learning_rate,
                    config->weight_decay,
                    config->beta1,
                    config->beta2,
                    config->epsilon
                )) {
                    goto training_failed;
                }
                total_training_steps++;
                epoch_samples += (uint64_t)actual_batch;
                epoch_loss += (double)loss * (double)actual_batch;
                if (config->log_every > 0 && total_training_steps % (uint64_t)config->log_every == 0) {
                    printf("Step %" PRIu64 ", loss %f\n", total_training_steps, loss);
                    fflush(stdout);
                }
            }
        }

        printf(
            "Epoch %d, loss %f\n",
            epoch + 1,
            epoch_samples > 0 ? (float)(epoch_loss / (double)epoch_samples) : 0.0f
        );
        fflush(stdout);
    }

    free(corpus);
    free(blocks);
    free(batch_starts);
    free(input);
    free(target);
    free_workspace(model, full_workspace);
    return 1;

training_failed:
    free(corpus);
    free(blocks);
    free(batch_starts);
    free(input);
    free(target);
    free_workspace(model, full_workspace);
    return 0;
}

static int compare_token_scores(const void* left, const void* right) {
    const TokenScore* a = left;
    const TokenScore* b = right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->token - b->token;
}

static int sample_next_token(
    GPTModel* model,
    const float* hidden,
    float* logits,
    float temperature,
    int top_k,
    RandomState* random
) {
    int d = model->embedding_dim;
    int vocab = model->vocab_size;
    int best_token = 0;
    float best_score = -FLT_MAX;

    for (int token = 0; token < vocab; token++) {
        const float* embedding = model->token_embedding->data + (size_t)token * (size_t)d;
        float score = 0.0f;
        for (int i = 0; i < d; i++) {
            score += hidden[i] * embedding[i];
        }
        if (!isfinite(score)) {
            return -1;
        }
        logits[token] = score;
        if (score > best_score) {
            best_score = score;
            best_token = token;
        }
    }

    if (temperature <= 0.0f || top_k == 1) {
        return best_token;
    }

    int candidate_count = top_k <= 0 || top_k > vocab ? vocab : top_k;
    size_t candidate_bytes = 0;
    if (!checked_size_multiply((size_t)vocab, sizeof(TokenScore), &candidate_bytes)) {
        return -1;
    }
    TokenScore* candidates = malloc(candidate_bytes);
    if (candidates == NULL) {
        return -1;
    }
    for (int token = 0; token < vocab; token++) {
        candidates[token].score = logits[token];
        candidates[token].token = token;
    }
    qsort(candidates, (size_t)vocab, sizeof(TokenScore), compare_token_scores);

    float maximum = candidates[0].score / temperature;
    double total = 0.0;
    for (int i = 0; i < candidate_count; i++) {
        double probability = exp((double)candidates[i].score / (double)temperature - (double)maximum);
        logits[i] = (float)probability;
        total += probability;
    }

    double pick = (double)random_uniform(random) * total;
    double cumulative = 0.0;
    int selected = candidates[candidate_count - 1].token;
    for (int i = 0; i < candidate_count; i++) {
        cumulative += logits[i];
        if (pick <= cumulative) {
            selected = candidates[i].token;
            break;
        }
    }
    free(candidates);
    return selected;
}

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
) {
    if (model == NULL || tokenizer == NULL || prompt == NULL || prompt_length <= 0 ||
        max_new_tokens < 0 || output_length == NULL || !isfinite(temperature) ||
        temperature < 0.0f || top_k < 0 || !parameters_are_finite(model) ||
        !gpt_bind_tokenizer(model, tokenizer)) {
        return NULL;
    }

    int prompt_tokens = bpe_encode(tokenizer, prompt, prompt_length, NULL, 0);
    if (prompt_tokens <= 0 || prompt_tokens > INT_MAX - max_new_tokens) {
        return NULL;
    }

    int capacity = prompt_tokens + max_new_tokens;
    size_t token_bytes = 0;
    if (!checked_size_multiply((size_t)capacity, sizeof(int32_t), &token_bytes)) {
        return NULL;
    }
    int32_t* tokens = malloc(token_bytes);
    if (tokens == NULL || bpe_encode(tokenizer, prompt, prompt_length, tokens, prompt_tokens) != prompt_tokens) {
        free(tokens);
        return NULL;
    }

    RandomState random = {seed == 0 ? 1u : seed};
    int count = prompt_tokens;
    for (int generated = 0; generated < max_new_tokens; generated++) {
        int sequence = count < model->context_length ? count : model->context_length;
        const int32_t* context_tokens = tokens + count - sequence;
        GPTWorkspace* workspace = create_workspace(model, 1, sequence);
        if (workspace == NULL || !forward_transformer(model, context_tokens, workspace)) {
            free_workspace(model, workspace);
            free(tokens);
            return NULL;
        }
        const float* hidden = workspace->final_norm +
            (size_t)(sequence - 1) * (size_t)model->embedding_dim;
        int next = sample_next_token(
            model,
            hidden,
            workspace->logits,
            temperature,
            top_k,
            &random
        );
        free_workspace(model, workspace);
        if (next < 0 || next >= model->vocab_size) {
            free(tokens);
            return NULL;
        }
        tokens[count++] = next;
    }

    int bytes = bpe_decode(tokenizer, tokens, count, NULL, 0);
    if (bytes < 0) {
        free(tokens);
        return NULL;
    }
    unsigned char* result = malloc((size_t)bytes + 1);
    if (result == NULL || bpe_decode(tokenizer, tokens, count, result, bytes) != bytes) {
        free(result);
        free(tokens);
        return NULL;
    }
    result[bytes] = 0;
    *output_length = bytes;
    free(tokens);
    return result;
}

int gpt_save(GPTModel* model, const char* path) {
    if (model == NULL || path == NULL || !parameters_are_finite(model)) {
        return 0;
    }
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }

    static const unsigned char magic[8] = {'N', 'N', 'G', 'P', 'T', '2', '\r', '\n'};
    int32_t config[6] = {
        model->vocab_size,
        model->context_length,
        model->embedding_dim,
        model->heads,
        model->layers,
        model->hidden_dim
    };
    uint64_t step = model->optimizer_step;
    uint64_t fingerprint = model->tokenizer_fingerprint;
    int valid = fwrite(magic, 1, sizeof(magic), file) == sizeof(magic) &&
        fwrite(config, sizeof(int32_t), 6, file) == 6 &&
        fwrite(&step, sizeof(step), 1, file) == 1 &&
        fwrite(&fingerprint, sizeof(fingerprint), 1, file) == 1;

    for (int p = 0; valid && p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        valid = fwrite(parameter->data, sizeof(float), parameter->count, file) == parameter->count &&
            fwrite(parameter->first_moment, sizeof(float), parameter->count, file) == parameter->count &&
            fwrite(parameter->second_moment, sizeof(float), parameter->count, file) == parameter->count;
    }
    if (fclose(file) != 0) {
        valid = 0;
    }
    return valid;
}

GPTModel* gpt_load(const char* path) {
    if (path == NULL) {
        return NULL;
    }
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    static const unsigned char version1[8] = {'N', 'N', 'G', 'P', 'T', '1', '\r', '\n'};
    static const unsigned char version2[8] = {'N', 'N', 'G', 'P', 'T', '2', '\r', '\n'};
    unsigned char magic[8];
    int32_t config[6];
    uint64_t step = 0;
    uint64_t fingerprint = 0;
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic)) {
        fclose(file);
        return NULL;
    }
    int is_version1 = memcmp(magic, version1, sizeof(magic)) == 0;
    int is_version2 = memcmp(magic, version2, sizeof(magic)) == 0;
    if ((!is_version1 && !is_version2) ||
        fread(config, sizeof(int32_t), 6, file) != 6 ||
        fread(&step, sizeof(step), 1, file) != 1 ||
        (is_version2 && fread(&fingerprint, sizeof(fingerprint), 1, file) != 1)) {
        fclose(file);
        return NULL;
    }

    GPTModel* model = create_gpt_model(
        config[0], config[1], config[2], config[3], config[4], config[5], 1
    );
    if (model == NULL) {
        fclose(file);
        return NULL;
    }
    model->optimizer_step = step;
    model->tokenizer_fingerprint = fingerprint;

    int valid = 1;
    for (int p = 0; valid && p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        valid = fread(parameter->data, sizeof(float), parameter->count, file) == parameter->count &&
            fread(parameter->first_moment, sizeof(float), parameter->count, file) == parameter->count &&
            fread(parameter->second_moment, sizeof(float), parameter->count, file) == parameter->count;
    }
    if (valid && (fgetc(file) != EOF || ferror(file))) {
        valid = 0;
    }
    if (fclose(file) != 0) {
        valid = 0;
    }
    if (!valid || !parameters_are_finite(model)) {
        free_gpt_model(model);
        return NULL;
    }
    return model;
}

void gpt_free_bytes(unsigned char* bytes) {
    free(bytes);
}

void free_gpt_model(GPTModel* model) {
    if (model == NULL) {
        return;
    }

    if (model->blocks != NULL) {
        for (int layer = 0; layer < model->layers; layer++) {
            GPTBlock* block = &model->blocks[layer];
            block->rms_attention = NULL;
            block->qkv = NULL;
            block->attention_output = NULL;
            block->rms_feed_forward = NULL;
            block->feed_forward_gate = NULL;
            block->feed_forward_value = NULL;
            block->feed_forward_output = NULL;
        }
    }

    for (int p = 0; p < model->parameter_count; p++) {
        free_parameter(model->parameters[p]);
    }
    free(model->parameters);
    free(model->blocks);
    free(model);
}
