#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "transformer_internal.h"

uint32_t random_next(RandomState* random) {
    random->state = random->state * 1664525u + 1013904223u;
    return random->state;
}

int random_bounded(RandomState* random, int limit) {
    if (limit <= 1) {
        return 0;
    }
    return (int)(random_next(random) % (uint32_t)limit);
}

float random_uniform(RandomState* random) {
    return (float)(random_next(random) >> 8) / 16777216.0f;
}

static Parameter* create_parameter(size_t count, int weight_decay) {
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }

    Parameter* parameter = calloc(1, sizeof(Parameter));
    if (parameter == NULL) {
        return NULL;
    }

    parameter->count = count;
    parameter->weight_decay = weight_decay;
    parameter->data = malloc(count * sizeof(float));
    parameter->grad = calloc(count, sizeof(float));
    parameter->first_moment = calloc(count, sizeof(float));
    parameter->second_moment = calloc(count, sizeof(float));

    if (parameter->data == NULL || parameter->grad == NULL ||
        parameter->first_moment == NULL || parameter->second_moment == NULL) {
        free(parameter->data);
        free(parameter->grad);
        free(parameter->first_moment);
        free(parameter->second_moment);
        free(parameter);
        return NULL;
    }
    return parameter;
}

void free_parameter(Parameter* parameter) {
    if (parameter == NULL) {
        return;
    }
    free(parameter->data);
    free(parameter->grad);
    free(parameter->first_moment);
    free(parameter->second_moment);
    free(parameter);
}

static int register_parameter(GPTModel* model, Parameter* parameter) {
    if (model->parameter_count == model->parameter_capacity) {
        if (model->parameter_capacity > INT_MAX / 2) {
            return 0;
        }
        int capacity = model->parameter_capacity == 0 ? 16 : model->parameter_capacity * 2;
        Parameter** resized = realloc(
            model->parameters,
            (size_t)capacity * sizeof(Parameter*)
        );
        if (resized == NULL) {
            return 0;
        }
        model->parameters = resized;
        model->parameter_capacity = capacity;
    }
    model->parameters[model->parameter_count++] = parameter;
    return 1;
}

Parameter* add_parameter(GPTModel* model, size_t count, int weight_decay) {
    Parameter* parameter = create_parameter(count, weight_decay);
    if (parameter == NULL || !register_parameter(model, parameter)) {
        free_parameter(parameter);
        return NULL;
    }
    return parameter;
}

void initialize_matrix(Parameter* parameter, int input_size, int output_size, RandomState* random) {
    float limit = sqrtf(6.0f / ((float)input_size + (float)output_size));
    for (size_t i = 0; i < parameter->count; i++) {
        parameter->data[i] = (2.0f * random_uniform(random) - 1.0f) * limit;
    }
}

void initialize_norm(Parameter* parameter) {
    for (size_t i = 0; i < parameter->count; i++) {
        parameter->data[i] = 1.0f;
    }
}

void zero_parameter_gradients(GPTModel* model) {
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        memset(parameter->grad, 0, parameter->count * sizeof(float));
    }
}

float gradient_norm(GPTModel* model) {
    if (model == NULL) {
        return NAN;
    }

    /* Stable scaled sum-of-squares, equivalent to BLAS nrm2 without a dependency. */
    double scale = 0.0;
    double sum_squares = 1.0;
    int has_nonzero = 0;

    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            double value = fabs((double)parameter->grad[i]);
            if (!isfinite(value)) {
                return NAN;
            }
            if (value == 0.0) {
                continue;
            }
            has_nonzero = 1;
            if (scale < value) {
                double ratio = scale / value;
                sum_squares = 1.0 + sum_squares * ratio * ratio;
                scale = value;
            } else {
                double ratio = value / scale;
                sum_squares += ratio * ratio;
            }
        }
    }

    if (!has_nonzero) {
        return 0.0f;
    }
    double norm = scale * sqrt(sum_squares);
    return norm > (double)FLT_MAX ? INFINITY : (float)norm;
}

void scale_gradients(GPTModel* model, float scale) {
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            parameter->grad[i] *= scale;
        }
    }
}

int gradients_are_finite(GPTModel* model) {
    if (model == NULL) {
        return 0;
    }
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            if (!isfinite(parameter->grad[i])) {
                return 0;
            }
        }
    }
    return 1;
}

int parameters_are_finite(GPTModel* model) {
    if (model == NULL) {
        return 0;
    }
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            if (!isfinite(parameter->data[i]) ||
                !isfinite(parameter->first_moment[i]) ||
                !isfinite(parameter->second_moment[i])) {
                return 0;
            }
        }
    }
    return 1;
}

int adamw_step(
    GPTModel* model,
    float learning_rate,
    float weight_decay,
    float beta1,
    float beta2,
    float epsilon
) {
    if (model == NULL || model->optimizer_step == UINT64_MAX ||
        !isfinite(learning_rate) || learning_rate <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(epsilon) || epsilon <= 0.0f ||
        !gradients_are_finite(model) || !parameters_are_finite(model)) {
        return 0;
    }

    uint64_t next_step = model->optimizer_step + 1;
    double correction1 = 1.0 - pow((double)beta1, (double)next_step);
    double correction2 = 1.0 - pow((double)beta2, (double)next_step);
    if (!isfinite(correction1) || !isfinite(correction2) ||
        correction1 <= 0.0 || correction2 <= 0.0) {
        return 0;
    }

    /* Validate the complete update before mutating any optimizer or parameter state. */
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            float gradient = parameter->grad[i];
            float first = beta1 * parameter->first_moment[i] + (1.0f - beta1) * gradient;
            float second = beta2 * parameter->second_moment[i] +
                (1.0f - beta2) * gradient * gradient;
            float first_hat = (float)((double)first / correction1);
            float second_hat = (float)((double)second / correction2);
            if (!isfinite(first) || !isfinite(second) || second_hat < 0.0f ||
                !isfinite(first_hat) || !isfinite(second_hat)) {
                return 0;
            }
            float update = first_hat / (sqrtf(second_hat) + epsilon);
            if (parameter->weight_decay) {
                update += weight_decay * parameter->data[i];
            }
            float next_value = parameter->data[i] - learning_rate * update;
            if (!isfinite(update) || !isfinite(next_value)) {
                return 0;
            }
        }
    }

    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            float gradient = parameter->grad[i];
            float first = beta1 * parameter->first_moment[i] + (1.0f - beta1) * gradient;
            float second = beta2 * parameter->second_moment[i] +
                (1.0f - beta2) * gradient * gradient;
            parameter->first_moment[i] = first;
            parameter->second_moment[i] = second;

            float first_hat = (float)((double)first / correction1);
            float second_hat = (float)((double)second / correction2);
            float update = first_hat / (sqrtf(second_hat) + epsilon);
            if (parameter->weight_decay) {
                update += weight_decay * parameter->data[i];
            }
            parameter->data[i] -= learning_rate * update;
        }
    }

    model->optimizer_step = next_step;
    return 1;
}

void rms_norm_forward(
    const float* input,
    const float* weight,
    float* output,
    float* inverse_rms,
    size_t rows,
    int width
) {
    for (size_t row = 0; row < rows; row++) {
        const float* x = input + row * (size_t)width;
        float* y = output + row * (size_t)width;
        double squares = 0.0;
        for (int i = 0; i < width; i++) {
            squares += (double)x[i] * (double)x[i];
        }
        float inverse = 1.0f / sqrtf((float)(squares / (double)width) + GPT_RMS_EPSILON);
        inverse_rms[row] = inverse;
        for (int i = 0; i < width; i++) {
            y[i] = x[i] * inverse * weight[i];
        }
    }
}

void rms_norm_backward(
    const float* input,
    const float* weight,
    const float* inverse_rms,
    const float* output_gradient,
    float* input_gradient,
    float* weight_gradient,
    size_t rows,
    int width
) {
    for (size_t row = 0; row < rows; row++) {
        const float* x = input + row * (size_t)width;
        const float* dy = output_gradient + row * (size_t)width;
        float* dx = input_gradient + row * (size_t)width;
        float inverse = inverse_rms[row];
        double dot = 0.0;

        for (int i = 0; i < width; i++) {
            float normalized = x[i] * inverse;
            weight_gradient[i] += dy[i] * normalized;
            dot += (double)(dy[i] * weight[i]) * (double)x[i];
        }

        float correction = inverse * inverse * inverse * (float)(dot / (double)width);
        for (int i = 0; i < width; i++) {
            float scaled_gradient = dy[i] * weight[i];
            dx[i] += inverse * scaled_gradient - x[i] * correction;
        }
    }
}

float silu(float value) {
    float sigmoid = 1.0f / (1.0f + expf(-value));
    return value * sigmoid;
}

float silu_derivative(float value) {
    float sigmoid = 1.0f / (1.0f + expf(-value));
    return sigmoid + value * sigmoid * (1.0f - sigmoid);
}

void rope_apply(float* qkv, int batch, int sequence, int embedding_dim, int heads, int inverse) {
    int head_dim = embedding_dim / heads;
    float direction = inverse ? -1.0f : 1.0f;

    for (int b = 0; b < batch; b++) {
        for (int position = 0; position < sequence; position++) {
            size_t row = (size_t)b * (size_t)sequence + (size_t)position;
            float* q = qkv + row * (size_t)(3 * embedding_dim);
            float* k = q + embedding_dim;

            for (int head = 0; head < heads; head++) {
                float* q_head = q + head * head_dim;
                float* k_head = k + head * head_dim;

                for (int pair = 0; pair < head_dim; pair += 2) {
                    float exponent = (float)pair / (float)head_dim;
                    float frequency = powf(GPT_ROPE_THETA, -exponent);
                    float angle = direction * (float)position * frequency;
                    float cosine = cosf(angle);
                    float sine = sinf(angle);

                    float q0 = q_head[pair];
                    float q1 = q_head[pair + 1];
                    q_head[pair] = q0 * cosine - q1 * sine;
                    q_head[pair + 1] = q0 * sine + q1 * cosine;

                    float k0 = k_head[pair];
                    float k1 = k_head[pair + 1];
                    k_head[pair] = k0 * cosine - k1 * sine;
                    k_head[pair + 1] = k0 * sine + k1 * cosine;
                }
            }
        }
    }
}

static size_t attention_probability_index(
    int batch_index,
    int head,
    int query,
    int key,
    int heads,
    int sequence
) {
    return ((((size_t)batch_index * (size_t)heads + (size_t)head) * (size_t)sequence +
        (size_t)query) * (size_t)sequence + (size_t)key);
}

void attention_forward(
    const float* qkv,
    float* probabilities,
    float* context,
    int batch,
    int sequence,
    int embedding_dim,
    int heads
) {
    int head_dim = embedding_dim / heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    memset(context, 0, (size_t)batch * (size_t)sequence * (size_t)embedding_dim * sizeof(float));

    for (int b = 0; b < batch; b++) {
        for (int head = 0; head < heads; head++) {
            for (int query = 0; query < sequence; query++) {
                size_t query_row = (size_t)b * (size_t)sequence + (size_t)query;
                const float* q = qkv + query_row * (size_t)(3 * embedding_dim) + head * head_dim;
                float maximum = -FLT_MAX;

                for (int key = 0; key <= query; key++) {
                    size_t key_row = (size_t)b * (size_t)sequence + (size_t)key;
                    const float* k = qkv + key_row * (size_t)(3 * embedding_dim) +
                        embedding_dim + head * head_dim;
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        score += q[i] * k[i];
                    }
                    score *= scale;
                    size_t probability_index = attention_probability_index(
                        b, head, query, key, heads, sequence
                    );
                    probabilities[probability_index] = score;
                    if (score > maximum) {
                        maximum = score;
                    }
                }

                float total = 0.0f;
                for (int key = 0; key <= query; key++) {
                    size_t probability_index = attention_probability_index(
                        b, head, query, key, heads, sequence
                    );
                    float value = expf(probabilities[probability_index] - maximum);
                    probabilities[probability_index] = value;
                    total += value;
                }
                for (int key = 0; key <= query; key++) {
                    size_t probability_index = attention_probability_index(
                        b, head, query, key, heads, sequence
                    );
                    probabilities[probability_index] /= total;
                }
                for (int key = query + 1; key < sequence; key++) {
                    probabilities[attention_probability_index(
                        b, head, query, key, heads, sequence
                    )] = 0.0f;
                }

                float* output = context + query_row * (size_t)embedding_dim + head * head_dim;
                for (int key = 0; key <= query; key++) {
                    size_t key_row = (size_t)b * (size_t)sequence + (size_t)key;
                    const float* v = qkv + key_row * (size_t)(3 * embedding_dim) +
                        2 * embedding_dim + head * head_dim;
                    float probability = probabilities[attention_probability_index(
                        b, head, query, key, heads, sequence
                    )];
                    for (int i = 0; i < head_dim; i++) {
                        output[i] += probability * v[i];
                    }
                }
            }
        }
    }
}

void attention_backward(
    const float* qkv,
    const float* probabilities,
    const float* context_gradient,
    float* qkv_gradient,
    float* scratch,
    int batch,
    int sequence,
    int embedding_dim,
    int heads
) {
    int head_dim = embedding_dim / heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    memset(
        qkv_gradient,
        0,
        (size_t)batch * (size_t)sequence * (size_t)(3 * embedding_dim) * sizeof(float)
    );

    for (int b = 0; b < batch; b++) {
        for (int head = 0; head < heads; head++) {
            for (int query = 0; query < sequence; query++) {
                size_t query_row = (size_t)b * (size_t)sequence + (size_t)query;
                const float* q = qkv + query_row * (size_t)(3 * embedding_dim) + head * head_dim;
                float* dq = qkv_gradient + query_row * (size_t)(3 * embedding_dim) + head * head_dim;
                const float* dcontext = context_gradient +
                    query_row * (size_t)embedding_dim + head * head_dim;

                float softmax_dot = 0.0f;
                for (int key = 0; key <= query; key++) {
                    size_t key_row = (size_t)b * (size_t)sequence + (size_t)key;
                    const float* v = qkv + key_row * (size_t)(3 * embedding_dim) +
                        2 * embedding_dim + head * head_dim;
                    float dp = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        dp += dcontext[i] * v[i];
                    }
                    scratch[key] = dp;
                    softmax_dot += probabilities[attention_probability_index(
                        b, head, query, key, heads, sequence
                    )] * dp;
                }

                for (int key = 0; key <= query; key++) {
                    size_t key_row = (size_t)b * (size_t)sequence + (size_t)key;
                    float probability = probabilities[attention_probability_index(
                        b, head, query, key, heads, sequence
                    )];
                    float dscore = probability * (scratch[key] - softmax_dot) * scale;

                    const float* k = qkv + key_row * (size_t)(3 * embedding_dim) +
                        embedding_dim + head * head_dim;
                    float* dk = qkv_gradient + key_row * (size_t)(3 * embedding_dim) +
                        embedding_dim + head * head_dim;
                    float* dv = qkv_gradient + key_row * (size_t)(3 * embedding_dim) +
                        2 * embedding_dim + head * head_dim;

                    for (int i = 0; i < head_dim; i++) {
                        dq[i] += dscore * k[i];
                        dk[i] += dscore * q[i];
                        dv[i] += probability * dcontext[i];
                    }
                }
            }
        }
    }
}
