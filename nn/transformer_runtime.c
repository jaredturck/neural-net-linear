#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "transformer_internal.h"

static int allocate_float(float** pointer, size_t count) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    *pointer = malloc(count * sizeof(float));
    return *pointer != NULL;
}

GPTWorkspace* create_workspace(GPTModel* model, int batch, int sequence) {
    if (model == NULL || batch <= 0 || sequence <= 0 || sequence > model->context_length) {
        return NULL;
    }

    GPTWorkspace* workspace = calloc(1, sizeof(GPTWorkspace));
    if (workspace == NULL) {
        return NULL;
    }
    workspace->batch = batch;
    workspace->sequence = sequence;
    workspace->rows = (size_t)batch * (size_t)sequence;

    size_t rows = workspace->rows;
    size_t state_count = (size_t)(model->layers + 1) * rows * (size_t)model->embedding_dim;
    size_t row_embedding = rows * (size_t)model->embedding_dim;
    size_t row_qkv = rows * (size_t)(3 * model->embedding_dim);
    size_t row_hidden = rows * (size_t)model->hidden_dim;

    workspace->blocks = calloc((size_t)model->layers, sizeof(BlockCache));
    if (workspace->blocks == NULL ||
        !allocate_float(&workspace->states, state_count) ||
        !allocate_float(&workspace->final_norm, row_embedding) ||
        !allocate_float(&workspace->final_inverse_rms, rows) ||
        !allocate_float(&workspace->gradient_a, row_embedding) ||
        !allocate_float(&workspace->gradient_b, row_embedding) ||
        !allocate_float(&workspace->gradient_c, row_embedding) ||
        !allocate_float(&workspace->gradient_qkv, row_qkv) ||
        !allocate_float(&workspace->gradient_hidden_a, row_hidden) ||
        !allocate_float(&workspace->gradient_hidden_b, row_hidden) ||
        !allocate_float(&workspace->gradient_hidden_c, row_hidden) ||
        !allocate_float(&workspace->attention_scratch, (size_t)sequence) ||
        !allocate_float(&workspace->logits, (size_t)model->vocab_size)) {
        goto fail;
    }

    for (int layer = 0; layer < model->layers; layer++) {
        BlockCache* cache = &workspace->blocks[layer];
        size_t probabilities = (size_t)batch * (size_t)model->heads *
            (size_t)sequence * (size_t)sequence;
        if (!allocate_float(&cache->norm_attention, row_embedding) ||
            !allocate_float(&cache->qkv, row_qkv) ||
            !allocate_float(&cache->attention_probabilities, probabilities) ||
            !allocate_float(&cache->attention_context, row_embedding) ||
            !allocate_float(&cache->after_attention, row_embedding) ||
            !allocate_float(&cache->norm_feed_forward, row_embedding) ||
            !allocate_float(&cache->feed_forward_gate, row_hidden) ||
            !allocate_float(&cache->feed_forward_value, row_hidden) ||
            !allocate_float(&cache->feed_forward_product, row_hidden) ||
            !allocate_float(&cache->inverse_rms_attention, rows) ||
            !allocate_float(&cache->inverse_rms_feed_forward, rows)) {
            goto fail;
        }
    }

    return workspace;

fail:
    if (workspace != NULL) {
        if (workspace->blocks != NULL) {
            for (int layer = 0; layer < model->layers; layer++) {
                BlockCache* cache = &workspace->blocks[layer];
                free(cache->norm_attention);
                free(cache->qkv);
                free(cache->attention_probabilities);
                free(cache->attention_context);
                free(cache->after_attention);
                free(cache->norm_feed_forward);
                free(cache->feed_forward_gate);
                free(cache->feed_forward_value);
                free(cache->feed_forward_product);
                free(cache->inverse_rms_attention);
                free(cache->inverse_rms_feed_forward);
            }
        }
        free(workspace->blocks);
        free(workspace->states);
        free(workspace->final_norm);
        free(workspace->final_inverse_rms);
        free(workspace->gradient_a);
        free(workspace->gradient_b);
        free(workspace->gradient_c);
        free(workspace->gradient_qkv);
        free(workspace->gradient_hidden_a);
        free(workspace->gradient_hidden_b);
        free(workspace->gradient_hidden_c);
        free(workspace->attention_scratch);
        free(workspace->logits);
        free(workspace);
    }
    return NULL;
}

void free_workspace(GPTModel* model, GPTWorkspace* workspace) {
    if (workspace == NULL) {
        return;
    }
    for (int layer = 0; layer < model->layers; layer++) {
        BlockCache* cache = &workspace->blocks[layer];
        free(cache->norm_attention);
        free(cache->qkv);
        free(cache->attention_probabilities);
        free(cache->attention_context);
        free(cache->after_attention);
        free(cache->norm_feed_forward);
        free(cache->feed_forward_gate);
        free(cache->feed_forward_value);
        free(cache->feed_forward_product);
        free(cache->inverse_rms_attention);
        free(cache->inverse_rms_feed_forward);
    }
    free(workspace->blocks);
    free(workspace->states);
    free(workspace->final_norm);
    free(workspace->final_inverse_rms);
    free(workspace->gradient_a);
    free(workspace->gradient_b);
    free(workspace->gradient_c);
    free(workspace->gradient_qkv);
    free(workspace->gradient_hidden_a);
    free(workspace->gradient_hidden_b);
    free(workspace->gradient_hidden_c);
    free(workspace->attention_scratch);
    free(workspace->logits);
    free(workspace);
}

static float* state_at(GPTModel* model, GPTWorkspace* workspace, int index) {
    return workspace->states + (size_t)index * workspace->rows * (size_t)model->embedding_dim;
}

int forward_transformer(
    GPTModel* model,
    const int32_t* tokens,
    GPTWorkspace* workspace
) {
    size_t rows = workspace->rows;
    int d = model->embedding_dim;
    float* first_state = state_at(model, workspace, 0);

    for (size_t row = 0; row < rows; row++) {
        int32_t token = tokens[row];
        if (token < 0 || token >= model->vocab_size) {
            return 0;
        }
        memcpy(
            first_state + row * (size_t)d,
            model->token_embedding->data + (size_t)token * (size_t)d,
            (size_t)d * sizeof(float)
        );
    }

    for (int layer = 0; layer < model->layers; layer++) {
        GPTBlock* block = &model->blocks[layer];
        BlockCache* cache = &workspace->blocks[layer];
        float* input = state_at(model, workspace, layer);
        float* output = state_at(model, workspace, layer + 1);

        rms_norm_forward(
            input,
            block->rms_attention->data,
            cache->norm_attention,
            cache->inverse_rms_attention,
            rows,
            d
        );
        matmul_forward(
            cache->norm_attention,
            block->qkv->data,
            cache->qkv,
            rows,
            d,
            3 * d
        );
        rope_apply(
            cache->qkv,
            workspace->batch,
            workspace->sequence,
            d,
            model->heads,
            0
        );
        attention_forward(
            cache->qkv,
            cache->attention_probabilities,
            cache->attention_context,
            workspace->batch,
            workspace->sequence,
            d,
            model->heads
        );
        matmul_forward(
            cache->attention_context,
            block->attention_output->data,
            cache->after_attention,
            rows,
            d,
            d
        );
        for (size_t i = 0; i < rows * (size_t)d; i++) {
            cache->after_attention[i] += input[i];
        }

        rms_norm_forward(
            cache->after_attention,
            block->rms_feed_forward->data,
            cache->norm_feed_forward,
            cache->inverse_rms_feed_forward,
            rows,
            d
        );
        matmul_forward(
            cache->norm_feed_forward,
            block->feed_forward_gate->data,
            cache->feed_forward_gate,
            rows,
            d,
            model->hidden_dim
        );
        matmul_forward(
            cache->norm_feed_forward,
            block->feed_forward_value->data,
            cache->feed_forward_value,
            rows,
            d,
            model->hidden_dim
        );
        for (size_t i = 0; i < rows * (size_t)model->hidden_dim; i++) {
            cache->feed_forward_product[i] =
                silu(cache->feed_forward_gate[i]) * cache->feed_forward_value[i];
        }
        matmul_forward(
            cache->feed_forward_product,
            block->feed_forward_output->data,
            output,
            rows,
            model->hidden_dim,
            d
        );
        for (size_t i = 0; i < rows * (size_t)d; i++) {
            output[i] += cache->after_attention[i];
        }
    }

    rms_norm_forward(
        state_at(model, workspace, model->layers),
        model->final_rms->data,
        workspace->final_norm,
        workspace->final_inverse_rms,
        rows,
        d
    );
    return 1;
}

static float output_loss_and_gradient(
    GPTModel* model,
    const int32_t* targets,
    GPTWorkspace* workspace,
    float* final_norm_gradient
) {
    int d = model->embedding_dim;
    int vocab = model->vocab_size;
    size_t rows = workspace->rows;
    double total_loss = 0.0;
    float gradient_scale = 1.0f / (float)rows;
    memset(final_norm_gradient, 0, rows * (size_t)d * sizeof(float));

    for (size_t row = 0; row < rows; row++) {
        const float* hidden = workspace->final_norm + row * (size_t)d;
        int32_t target = targets[row];
        if (target < 0 || target >= vocab) {
            return NAN;
        }

        float maximum = -FLT_MAX;
        for (int token = 0; token < vocab; token++) {
            const float* embedding = model->token_embedding->data + (size_t)token * (size_t)d;
            float logit = 0.0f;
            for (int i = 0; i < d; i++) {
                logit += hidden[i] * embedding[i];
            }
            workspace->logits[token] = logit;
            if (logit > maximum) {
                maximum = logit;
            }
        }

        double sum = 0.0;
        for (int token = 0; token < vocab; token++) {
            sum += exp((double)workspace->logits[token] - (double)maximum);
        }
        double log_sum_exp = (double)maximum + log(sum);
        total_loss += log_sum_exp - (double)workspace->logits[target];

        float* dhidden = final_norm_gradient + row * (size_t)d;
        for (int token = 0; token < vocab; token++) {
            float probability = (float)(exp((double)workspace->logits[token] -
                (double)maximum) / sum);
            float gradient = (probability - (token == target ? 1.0f : 0.0f)) * gradient_scale;
            float* embedding_gradient = model->token_embedding->grad + (size_t)token * (size_t)d;
            const float* embedding = model->token_embedding->data + (size_t)token * (size_t)d;
            for (int i = 0; i < d; i++) {
                embedding_gradient[i] += gradient * hidden[i];
                dhidden[i] += gradient * embedding[i];
            }
        }
    }

    return (float)(total_loss / (double)rows);
}

float train_batch(
    GPTModel* model,
    const int32_t* input_tokens,
    const int32_t* target_tokens,
    GPTWorkspace* workspace
) {
    int d = model->embedding_dim;
    size_t rows = workspace->rows;
    size_t embedding_values = rows * (size_t)d;
    size_t hidden_values = rows * (size_t)model->hidden_dim;
    size_t qkv_values = rows * (size_t)(3 * d);

    zero_parameter_gradients(model);
    if (!forward_transformer(model, input_tokens, workspace)) {
        return NAN;
    }

    float loss = output_loss_and_gradient(
        model,
        target_tokens,
        workspace,
        workspace->gradient_a
    );
    if (!isfinite(loss)) {
        return loss;
    }

    memset(workspace->gradient_b, 0, embedding_values * sizeof(float));
    rms_norm_backward(
        state_at(model, workspace, model->layers),
        model->final_rms->data,
        workspace->final_inverse_rms,
        workspace->gradient_a,
        workspace->gradient_b,
        model->final_rms->grad,
        rows,
        d
    );

    float* current_gradient = workspace->gradient_b;
    float* next_gradient = workspace->gradient_a;

    for (int layer = model->layers - 1; layer >= 0; layer--) {
        GPTBlock* block = &model->blocks[layer];
        BlockCache* cache = &workspace->blocks[layer];
        float* input = state_at(model, workspace, layer);

        /* Feed-forward residual: d(after_attention) starts with residual gradient. */
        memcpy(next_gradient, current_gradient, embedding_values * sizeof(float));
        memset(workspace->gradient_hidden_a, 0, hidden_values * sizeof(float));
        matmul_backward_accumulate(
            cache->feed_forward_product,
            block->feed_forward_output->data,
            current_gradient,
            workspace->gradient_hidden_a,
            block->feed_forward_output->grad,
            rows,
            model->hidden_dim,
            d
        );

        for (size_t i = 0; i < hidden_values; i++) {
            float gate = cache->feed_forward_gate[i];
            float value = cache->feed_forward_value[i];
            float gradient = workspace->gradient_hidden_a[i];
            workspace->gradient_hidden_b[i] = gradient * value * silu_derivative(gate);
            workspace->gradient_hidden_c[i] = gradient * silu(gate);
        }

        memset(workspace->gradient_c, 0, embedding_values * sizeof(float));
        matmul_backward_accumulate(
            cache->norm_feed_forward,
            block->feed_forward_gate->data,
            workspace->gradient_hidden_b,
            workspace->gradient_c,
            block->feed_forward_gate->grad,
            rows,
            d,
            model->hidden_dim
        );
        matmul_backward_accumulate(
            cache->norm_feed_forward,
            block->feed_forward_value->data,
            workspace->gradient_hidden_c,
            workspace->gradient_c,
            block->feed_forward_value->grad,
            rows,
            d,
            model->hidden_dim
        );

        memset(current_gradient, 0, embedding_values * sizeof(float));
        rms_norm_backward(
            cache->after_attention,
            block->rms_feed_forward->data,
            cache->inverse_rms_feed_forward,
            workspace->gradient_c,
            current_gradient,
            block->rms_feed_forward->grad,
            rows,
            d
        );
        for (size_t i = 0; i < embedding_values; i++) {
            next_gradient[i] += current_gradient[i];
        }

        /* Attention residual: d(input) starts with d(after_attention). */
        memcpy(current_gradient, next_gradient, embedding_values * sizeof(float));
        memset(workspace->gradient_c, 0, embedding_values * sizeof(float));
        matmul_backward_accumulate(
            cache->attention_context,
            block->attention_output->data,
            next_gradient,
            workspace->gradient_c,
            block->attention_output->grad,
            rows,
            d,
            d
        );

        attention_backward(
            cache->qkv,
            cache->attention_probabilities,
            workspace->gradient_c,
            workspace->gradient_qkv,
            workspace->attention_scratch,
            workspace->batch,
            workspace->sequence,
            d,
            model->heads
        );
        rope_apply(
            workspace->gradient_qkv,
            workspace->batch,
            workspace->sequence,
            d,
            model->heads,
            1
        );

        memset(workspace->gradient_c, 0, embedding_values * sizeof(float));
        matmul_backward_accumulate(
            cache->norm_attention,
            block->qkv->data,
            workspace->gradient_qkv,
            workspace->gradient_c,
            block->qkv->grad,
            rows,
            d,
            3 * d
        );

        memset(next_gradient, 0, embedding_values * sizeof(float));
        rms_norm_backward(
            input,
            block->rms_attention->data,
            cache->inverse_rms_attention,
            workspace->gradient_c,
            next_gradient,
            block->rms_attention->grad,
            rows,
            d
        );
        for (size_t i = 0; i < embedding_values; i++) {
            current_gradient[i] += next_gradient[i];
        }

        float* swap = current_gradient;
        current_gradient = next_gradient;
        next_gradient = swap;
        memcpy(current_gradient, next_gradient, embedding_values * sizeof(float));
    }

    for (size_t row = 0; row < rows; row++) {
        int32_t token = input_tokens[row];
        float* embedding_gradient = model->token_embedding->grad + (size_t)token * (size_t)d;
        const float* dx = current_gradient + row * (size_t)d;
        for (int i = 0; i < d; i++) {
            embedding_gradient[i] += dx[i];
        }
    }

    (void)qkv_values;
    return loss;
}
