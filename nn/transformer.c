#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
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

static uint32_t random_next(RandomState* random) {
    random->state = random->state * 1664525u + 1013904223u;
    return random->state;
}

static int random_bounded(RandomState* random, int limit) {
    if (limit <= 1) {
        return 0;
    }
    return (int)(random_next(random) % (uint32_t)limit);
}

static float random_uniform(RandomState* random) {
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

static void free_parameter(Parameter* parameter) {
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

static Parameter* add_parameter(GPTModel* model, size_t count, int weight_decay) {
    Parameter* parameter = create_parameter(count, weight_decay);
    if (parameter == NULL || !register_parameter(model, parameter)) {
        free_parameter(parameter);
        return NULL;
    }
    return parameter;
}

static void initialize_matrix(Parameter* parameter, int input_size, int output_size, RandomState* random) {
    float limit = sqrtf(6.0f / (float)(input_size + output_size));
    for (size_t i = 0; i < parameter->count; i++) {
        parameter->data[i] = (2.0f * random_uniform(random) - 1.0f) * limit;
    }
}

static void initialize_norm(Parameter* parameter) {
    for (size_t i = 0; i < parameter->count; i++) {
        parameter->data[i] = 1.0f;
    }
}

static void zero_parameter_gradients(GPTModel* model) {
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        memset(parameter->grad, 0, parameter->count * sizeof(float));
    }
}

static float gradient_norm(GPTModel* model) {
    double total = 0.0;
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            double value = parameter->grad[i];
            total += value * value;
        }
    }
    return sqrtf((float)total);
}

static void scale_gradients(GPTModel* model, float scale) {
    for (int p = 0; p < model->parameter_count; p++) {
        Parameter* parameter = model->parameters[p];
        for (size_t i = 0; i < parameter->count; i++) {
            parameter->grad[i] *= scale;
        }
    }
}

static void adamw_step(
    GPTModel* model,
    float learning_rate,
    float weight_decay,
    float beta1,
    float beta2,
    float epsilon
) {
    model->optimizer_step++;
    double correction1 = 1.0 - pow((double)beta1, (double)model->optimizer_step);
    double correction2 = 1.0 - pow((double)beta2, (double)model->optimizer_step);

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
}

static void rms_norm_forward(
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
        float inverse = 1.0f / sqrtf((float)²È="25•ÑÕÉ¸€Àì)ô()ÍÑ…Ñ¥Œ¥¹Ð½µÁ…É•}Ñ½­•¹}Í½É•Ì¡½¹ÍÐÙ½¥¨±•™Ð°½¹ÍÐÙ½¥¨É¥¡Ð¤ì(€€€½¹ÍÐQ½­•¹M½É”¨„€ô±•™Ðì(€€€½¹ÍÐQ½­•¹M½É”¨ˆ€ôÉ¥¡Ðì(€€€¥˜€¡„´ùÍ½É”€ðˆ´ùÍ½É”¤É•ÑÕÉ¸€Äì(€€€¥˜€¡„´ùÍ½É”€øˆ´ùÍ½É”¤É•ÑÕÉ¸€´Äì(€€€É•ÑÕÉ¸„´ùÑ½­•¸€´ˆ´ùÑ½­•¸ì)ô()ÍÑ…Ñ¥Œ¥¹ÐÍ…µÁ±•}¹•áÑ}Ñ½­•¸ (€€€AQ5½‘•°¨µ½‘•°°(€€€½¹ÍÐ™±½…Ð¨¡¥‘‘•¸°(€€€™±½…Ð¨±½¥ÑÌ°(€€€™±½…ÐÑ•µÁ•É…ÑÕÉ”°(€€€¥¹ÐÑ½Á}¬°(€€€I…¹‘½µMÑ…Ñ”¨É…¹‘½´(¤ì(€€€¥¹Ð€ôµ½‘•°´ù•µ‰•‘‘¥¹}‘¥´ì(€€€¥¹ÐÙ½…ˆ€ôµ½‘•°´ùÙ½…‰}Í¥é”ì(€€€¥¹Ð‰•ÍÑ}Ñ½­•¸€ô€Àì(€€€™±½…Ð‰•ÍÑ}Í½É”€ô€µ1Q}5`ì((€€€™½È€¡¥¹ÐÑ½­•¸€ô€ÀìÑ½­•¸€ðÙ½…ˆìÑ½­•¸¬¬¤ì(€€€€€€€½¹ÍÐ™±½…Ð¨•µ‰•‘‘¥¹œ€ôµ½‘•°´ùÑ½­•¹}•µ‰•‘‘¥¹œ´ù‘…Ñ„€¬€¡Í¥é•}Ð¥Ñ½­•¸€¨€¡Í¥é•}Ð¥ì(€€€€€€€™±½…ÐÍ½É”€ô€À¸Á˜ì(€€€€€€€™½È€¡¥¹Ð¤€ô€Àì¤€ðì¤¬¬¤ì(€€€€€€€€€€€Í½É”€¬ô¡¥‘‘•¹m¥t€¨•µ‰•‘‘¥¹m¥tì(€€€€€€€ô(€€€€€€€±½¥ÑÍmÑ½­•¹t€ôÍ½É”ì(€€€€€€€¥˜€¡Í½É”€ø‰•ÍÑ}Í½É”¤ì(€€€€€€€€€€€‰•ÍÑ}Í½É”€ôÍ½É”ì(€€€€€€€€€€€‰•ÍÑ}Ñ½­•¸€ôÑ½­•¸ì(€€€€€€€ô(€€€ô((€€€¥˜€¡Ñ•µÁ•É…ÑÕÉ”€ðô€À¸Á˜ñðÑ½Á}¬€ôô€Ä¤ì(€€€€€€€É•ÑÕÉ¸‰•ÍÑ}Ñ½­•¸ì(€€€ô((€€€¥¹Ð…¹‘¥‘…Ñ•}½Õ¹Ð€ôÑ½Á}¬€ðô€ÀñðÑ½Á}¬€øÙ½…ˆ€üÙ½…ˆ€èÑ½Á}¬ì(€€€Q½­•¹M½É”¨…¹‘¥‘…Ñ•Ì€ôµ…±±½Œ ¡Í¥é•}Ð¥Ù½…ˆ€¨Í¥é•½˜¡Q½­•¹M½É”¤¤ì(€€€¥˜€¡…¹‘¥‘…Ñ•Ì€ôô9U10¤ì(€€€€€€€É•ÑÕÉ¸‰•ÍÑ}Ñ½­•¸ì(€€€ô(€€€™½È€¡¥¹ÐÑ½­•¸€ô€ÀìÑ½­•¸€ðÙ½…ˆìÑ½­•¸¬¬¤ì(€€€€€€€…¹‘¥‘…Ñ•ÍmÑ½­•¹t¹Í½É”€ô±½¥ÑÍmÑ½­•¹tì(€€€€€€€…¹‘¥‘…Ñ•ÍmÑ½­•¹t¹Ñ½­•¸€ôÑ½­•¸ì(€€€ô(€€€ÅÍ½ÉÐ¡…¹‘¥‘…Ñ•Ì°€¡Í¥é•}Ð¥Ù½…ˆ°Í¥é•½˜¡Q½­•¹M½É”¤°½µÁ…É•}Ñ½­•¹}Í½É•Ì¤ì((€€€™±½…Ðµ…á¥µÕ´€ô…¹‘¥‘…Ñ•ÍlÁt¹Í½É”€¼Ñ•µÁ•É…ÑÕÉ”ì(€€€‘½Õ‰±”Ñ½Ñ…°€ô€À¸Àì(€€€™½È€¡¥¹Ð¤€ô€Àì¤€ð…¹‘¥‘…Ñ•}½Õ¹Ðì¤¬¬¤ì(€€€€€€€‘½Õ‰±”ÁÉ½‰…‰¥±¥Ñä€ô•áÀ ¡‘½Õ‰±”¥…¹‘¥‘…Ñ•Ím¥t¹Í½É”€¼€¡‘½Õ‰±”¥Ñ•µÁ•É…ÑÕÉ”€´€¡‘½Õ‰±”¥µ…á¥µÕ´¤ì(€€€€€€€±½¥ÑÍm¥t€ô€¡™±½…Ð¥ÁÉ½‰…‰¥±¥Ñäì(€€€€€€€Ñ½Ñ…°€¬ôÁÉ½‰…‰¥±¥Ñäì(€€€ô((€€€‘½Õ‰±”Á¥¬€ô€¡‘½Õ‰±”¥É…¹‘½µ}Õ¹¥™½É´¡É…¹‘½´¤€¨Ñ½Ñ…°ì(€€€‘½Õ‰±”ÕµÕ±…Ñ¥Ù”€ô€À¸Àì(€€€¥¹ÐÍ•±•Ñ•€ô…¹‘¥‘…Ñ•Ím…¹‘¥‘…Ñ•}½Õ¹Ð€´€Åt¹Ñ½­•¸ì(€€€™½È€¡¥¹Ð¤€ô€Àì¤€ð…¹‘¥‘…Ñ•}½Õ¹Ðì¤¬¬¤ì(€€€€€€€ÕµÕ±…Ñ¥Ù”€¬ô±½¥ÑÍm¥tì(€€€€€€€¥˜€¡Á¥¬€ðôÕµÕ±…Ñ¥Ù”¤ì(€€€€€€€€€€€Í•±•Ñ•€ô…¹‘¥‘…Ñ•Ím¥t¹Ñ½­•¸ì(€€€€€€€€€€€‰É•…¬ì(€€€€€€€ô(€€€ô(€€€™É•”¡…¹‘¥‘…Ñ•Ì¤ì(€€€É•ÑÕÉ¸Í•±•Ñ•ì)ô()Õ¹Í¥¹•¡…È¨ÁÑ}•¹•É…Ñ” (€€€AQ5½‘•°¨µ½‘•°°(€€€	AQ½­•¹¥é•È¨Ñ½­•¹¥é•È°(€€€½¹ÍÐÕ¹Í¥¹•¡…È¨ÁÉ½µÁÐ°(€€€¥¹ÐÁÉ½µÁÑ}±•¹Ñ °(€€€¥¹Ðµ…á}¹•Ý}Ñ½­•¹Ì°(€€€™±½…ÐÑ•µÁ•É…ÑÕÉ”°(€€€¥¹ÐÑ½Á}¬°(€€€Õ¹Í¥¹•¥¹ÐÍ••°(€€€¥¹Ð¨½ÕÑÁÕÑ}±•¹Ñ (¤ì(€€€¥˜€¡µ½‘•°€ôô9U10ñðÑ½­•¹¥é•È€ôô9U10ñðÁÉ½µÁÐ€ôô9U10ñðÁÉ½µÁÑ}±•¹Ñ €ðô€Àñð(€€€€€€€µ…á}¹•Ý}Ñ½­•¹Ì€ð€Àñð½ÕÑÁÕÑ}±•¹Ñ €ôô9U10ñð(€€€€€€€‰Á•}Ù½…‰}Í¥é”¡Ñ½­•¹¥é•È¤€„ôµ½‘•°´ùÙ½…‰}Í¥é”¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô((€€€¥¹ÐÁÉ½µÁÑ}Ñ½­•¹Ì€ô‰Á•}•¹½‘”¡Ñ½­•¹¥é•È°ÁÉ½µÁÐ°ÁÉ½µÁÑ}±•¹Ñ °9U10°€À¤ì(€€€¥˜€¡ÁÉ½µÁÑ}Ñ½­•¹Ì€ðô€ÀñðÁÉ½µÁÑ}Ñ½­•¹Ì€ø%9Q}5`€´µ…á}¹•Ý}Ñ½­•¹Ì¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô((€€€¥¹Ð…Á…¥Ñä€ôÁÉ½µÁÑ}Ñ½­•¹Ì€¬µ…á}¹•Ý}Ñ½­•¹Ìì(€€€¥¹ÐÌÉ}Ð¨Ñ½­•¹Ì€ôµ…±±½Œ ¡Í¥é•}Ð¥…Á…¥Ñä€¨Í¥é•½˜¡¥¹ÐÌÉ}Ð¤¤ì(€€€¥˜€¡Ñ½­•¹Ì€ôô9U10ñð‰Á•}•¹½‘”¡Ñ½­•¹¥é•È°ÁÉ½µÁÐ°ÁÉ½µÁÑ}±•¹Ñ °Ñ½­•¹Ì°ÁÉ½µÁÑ}Ñ½­•¹Ì¤€„ôÁÉ½µÁÑ}Ñ½­•¹Ì¤ì(€€€€€€€™É•”¡Ñ½­•¹Ì¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô((€€€I…¹‘½µMÑ…Ñ”É…¹‘½´€ôíÍ••€ôô€À€ü€ÅÔ€èÍ••‘ôì(€€€¥¹Ð½Õ¹Ð€ôÁÉ½µÁÑ}Ñ½­•¹Ìì(€€€™½È€¡¥¹Ð•¹•É…Ñ•€ô€Àì•¹•É…Ñ•€ðµ…á}¹•Ý}Ñ½­•¹Ìì•¹•É…Ñ•¬¬¤ì(€€€€€€€¥¹ÐÍ•ÅÕ•¹”€ô½Õ¹Ð€ðµ½‘•°´ù½¹Ñ•áÑ}±•¹Ñ €ü½Õ¹Ð€èµ½‘•°´ù½¹Ñ•áÑ}±•¹Ñ ì(€€€€€€€½¹ÍÐ¥¹ÐÌÉ}Ð¨½¹Ñ•áÑ}Ñ½­•¹Ì€ôÑ½­•¹Ì€¬½Õ¹Ð€´Í•ÅÕ•¹”ì(€€€€€€€AQ]½É­ÍÁ…”¨Ý½É­ÍÁ…”€ôÉ•…Ñ•}Ý½É­ÍÁ…”¡µ½‘•°°€Ä°Í•ÅÕ•¹”¤ì(€€€€€€€¥˜€¡Ý½É­ÍÁ…”€ôô9U10ñð€…™½ÉÝ…É‘}ÑÉ…¹Í™½Éµ•È¡µ½‘•°°½¹Ñ•áÑ}Ñ½­•¹Ì°Ý½É­ÍÁ…”¤¤ì(€€€€€€€€€€€™É••}Ý½É­ÍÁ…”¡µ½‘•°°Ý½É­ÍÁ…”¤ì(€€€€€€€€€€€™É•”¡Ñ½­•¹Ì¤ì(€€€€€€€€€€€É•ÑÕÉ¸9U10ì(€€€€€€€ô(€€€€€€€½¹ÍÐ™±½…Ð¨¡¥‘‘•¸€ôÝ½É­ÍÁ…”´ù™¥¹…±}¹½É´€¬(€€€€€€€€€€€€¡Í¥é•}Ð¤¡Í•ÅÕ•¹”€´€Ä¤€¨€¡Í¥é•}Ð¥µ½‘•°´ù•µ‰•‘‘¥¹}‘¥´ì(€€€€€€€¥¹Ð¹•áÐ€ôÍ…µÁ±•}¹•áÑ}Ñ½­•¸ (€€€€€€€€€€€µ½‘•°°(€€€€€€€€€€€¡¥‘‘•¸°(€€€€€€€€€€€Ý½É­ÍÁ…”´ù±½¥ÑÌ°(€€€€€€€€€€€Ñ•µÁ•É…ÑÕÉ”°(€€€€€€€€€€€Ñ½Á}¬°(€€€€€€€€€€€€™É…¹‘½´(€€€€€€€€¤ì(€€€€€€€™É••}Ý½É­ÍÁ…”¡µ½‘•°°Ý½É­ÍÁ…”¤ì(€€€€€€€Ñ½­•¹Ím½Õ¹Ð¬­t€ô¹•áÐì(€€€ô((€€€¥¹Ð‰åÑ•Ì€ô‰Á•}‘•½‘”¡Ñ½­•¹¥é•È°Ñ½­•¹Ì°½Õ¹Ð°9U10°€À¤ì(€€€¥˜€¡‰åÑ•Ì€ð€À¤ì(€€€€€€€™É•”¡Ñ½­•¹Ì¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô(€€€Õ¹Í¥¹•¡…È¨É•ÍÕ±Ð€ôµ…±±½Œ ¡Í¥é•}Ð¥‰åÑ•Ì€¬€Ä¤ì(€€€¥˜€¡É•ÍÕ±Ð€ôô9U10ñð‰Á•}‘•½‘”¡Ñ½­•¹¥é•È°Ñ½­•¹Ì°½Õ¹Ð°É•ÍÕ±Ð°‰åÑ•Ì¤€„ô‰åÑ•Ì¤ì(€€€€€€€™É•”¡É•ÍÕ±Ð¤ì(€€€€€€€™É•”¡Ñ½­•¹Ì¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô(€€€É•ÍÕ±Ñm‰åÑ•Ít€ô€Àì(€€€€©½ÕÑÁÕÑ}±•¹Ñ €ô‰åÑ•Ìì(€€€™É•”¡Ñ½­•¹Ì¤ì(€€€É•ÑÕÉ¸É•ÍÕ±Ðì)ô()¥¹ÐÁÑ}Í…Ù”¡AQ5½‘•°¨µ½‘•°°½¹ÍÐ¡…È¨Á…Ñ ¤ì(€€€¥˜€¡µ½‘•°€ôô9U10ñðÁ…Ñ €ôô9U10¤ì(€€€€€€€É•ÑÕÉ¸€Àì(€€€ô(€€€%1¨™¥±”€ô™½Á•¸¡Á…Ñ °€‰Ýˆˆ¤ì(€€€¥˜€¡™¥±”€ôô9U10¤ì(€€€€€€€É•ÑÕÉ¸€Àì(€€€ô((€€€ÍÑ…Ñ¥Œ½¹ÍÐÕ¹Í¥¹•¡…Èµ…¥lát€ôì8œ°€8œ°€œ°€@œ°€Pœ°€œÄœ°€qÈœ°€q¸ôì(€€€¥¹ÐÌÉ}Ð½¹™¥lÙt€ôì(€€€€€€€µ½‘•°´ùÙ½…‰}Í¥é”°(€€€€€€€µ½‘•°´ù½¹Ñ•áÑ}±•¹Ñ °(€€€€€€€µ½‘•°´ù•µ‰•‘‘¥¹}‘¥´°(€€€€€€€µ½‘•°´ù¡•…‘Ì°(€€€€€€€µ½‘•°´ù±…å•ÉÌ°(€€€€€€€µ½‘•°´ù¡¥‘‘•¹}‘¥´(€€€ôì(€€€Õ¥¹ÐØÑ}ÐÍÑ•À€ôµ½‘•°´ù½ÁÑ¥µ¥é•É}ÍÑ•Àì(€€€¥¹ÐÙ…±¥€ô™ÝÉ¥Ñ”¡µ…¥Œ°€Ä°Í¥é•½˜¡µ…¥Œ¤°™¥±”¤€ôôÍ¥é•½˜¡µ…¥Œ¤€˜˜(€€€€€€€™ÝÉ¥Ñ”¡½¹™¥œ°Í¥é•½˜¡¥¹ÐÌÉ}Ð¤°€Ø°™¥±”¤€ôô€Ø€˜˜(€€€€€€€™ÝÉ¥Ñ” ™ÍÑ•À°Í¥é•½˜¡ÍÑ•À¤°€Ä°™¥±”¤€ôô€Äì((€€€™½È€¡¥¹ÐÀ€ô€ÀìÙ…±¥€˜˜À€ðµ½‘•°´ùÁ…É…µ•Ñ•É}½Õ¹ÐìÀ¬¬¤ì(€€€€€€€A…É…µ•Ñ•È¨Á…É…µ•Ñ•È€ôµ½‘•°´ùÁ…É…µ•Ñ•ÉÍmÁtì(€€€€€€€Ù…±¥€ô™ÝÉ¥Ñ”¡Á…É…µ•Ñ•È´ù‘…Ñ„°Í¥é•½˜¡™±½…Ð¤°Á…É…µ•Ñ•È´ù½Õ¹Ð°™¥±”¤€ôôÁ…É…µ•Ñ•È´ù½Õ¹Ð€˜˜(€€€€€€€€€€€™ÝÉ¥Ñ”¡Á…É…µ•Ñ•È´ù™¥ÉÍÑ}µ½µ•¹Ð°Í¥é•½˜¡™±½…Ð¤°Á…É…µ•Ñ•È´ù½Õ¹Ð°™¥±”¤€ôôÁ…É…µ•Ñ•È´ù½Õ¹Ð€˜˜(€€€€€€€€€€€™ÝÉ¥Ñ”¡Á…É…µ•Ñ•È´ùÍ•½¹‘}µ½µ•¹Ð°Í¥é•½˜¡™±½…Ð¤°Á…É…µ•Ñ•È´ù½Õ¹Ð°™¥±”¤€ôôÁ…É…µ•Ñ•È´ù½Õ¹Ðì(€€€ô(€€€¥˜€¡™±½Í”¡™¥±”¤€„ô€À¤ì(€€€€€€€Ù…±¥€ô€Àì(€€€ô(€€€É•ÑÕÉ¸Ù…±¥ì)ô()AQ5½‘•°¨ÁÑ}±½…¡½¹ÍÐ¡…È¨Á…Ñ ¤ì(€€€¥˜€¡Á…Ñ €ôô9U10¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô(€€€%1¨™¥±”€ô™½Á•¸¡Á…Ñ °€‰Éˆˆ¤ì(€€€¥˜€¡™¥±”€ôô9U10¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô((€€€ÍÑ…Ñ¥Œ½¹ÍÐÕ¹Í¥¹•¡…È•áÁ•Ñ•‘lát€ôì8œ°€8œ°€œ°€@œ°€Pœ°€œÄœ°€qÈœ°€q¸ôì(€€€Õ¹Í¥¹•¡…Èµ…¥látì(€€€¥¹ÐÌÉ}Ð½¹™¥lÙtì(€€€Õ¥¹ÐØÑ}ÐÍÑ•À€ô€Àì(€€€¥˜€¡™É•…¡µ…¥Œ°€Ä°Í¥é•½˜¡µ…¥Œ¤°™¥±”¤€„ôÍ¥é•½˜¡µ…¥Œ¤ñð(€€€€€€€µ•µµÀ¡µ…¥Œ°•áÁ•Ñ•°Í¥é•½˜¡µ…¥Œ¤¤€„ô€Àñð(€€€€€€€™É•…¡½¹™¥œ°Í¥é•½˜¡¥¹ÐÌÉ}Ð¤°€Ø°™¥±”¤€„ô€Øñð(€€€€€€€™É•… ™ÍÑ•À°Í¥é•½˜¡ÍÑ•À¤°€Ä°™¥±”¤€„ô€Ä¤ì(€€€€€€€™±½Í”¡™¥±”¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô((€€€AQ5½‘•°¨µ½‘•°€ôÉ•…Ñ•}ÁÑ}µ½‘•° (€€€€€€€½¹™¥lÁt°½¹™¥lÅt°½¹™¥lÉt°½¹™¥lÍt°½¹™¥lÑt°½¹™¥lÕt°€Ä(€€€€¤ì(€€€¥˜€¡µ½‘•°€ôô9U10¤ì(€€€€€€€™±½Í”¡™¥±”¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô(€€€µ½‘•°´ù½ÁÑ¥µ¥é•É}ÍÑ•À€ôÍÑ•Àì((€€€¥¹ÐÙ…±¥€ô€Äì(€€€™½È€¡¥¹ÐÀ€ô€ÀìÙ…±¥€˜˜À€ðµ½‘•°´ùÁ…É…µ•Ñ•É}½Õ¹ÐìÀ¬¬¤ì(€€€€€€€A…É…µ•Ñ•È¨Á…É…µ•Ñ•È€ôµ½‘•°´ùÁ…É…µ•Ñ•ÉÍmÁtì(€€€€€€€Ù…±¥€ô™É•…¡Á…É…µ•Ñ•È´ù‘…Ñ„°Í¥é•½˜¡™±½…Ð¤°Á…É…µ•Ñ•È´ù½Õ¹Ð°™¥±”¤€ôôÁ…É…µ•Ñ•È´ù½Õ¹Ð€˜˜(€€€€€€€€€€€™É•…¡Á…É…µ•Ñ•È´ù™¥ÉÍÑ}µ½µ•¹Ð°Í¥é•½˜¡™±½…Ð¤°Á…É…µ•Ñ•È´ù½Õ¹Ð°™¥±”¤€ôôÁ…É…µ•Ñ•È´ù½Õ¹Ð€˜˜(€€€€€€€€€€€™É•…¡Á…É…µ•Ñ•È´ùÍ•½¹‘}µ½µ•¹Ð°Í¥é•½˜¡™±½…Ð¤°Á…É…µ•Ñ•È´ù½Õ¹Ð°™¥±”¤€ôôÁ…É…µ•Ñ•È´ù½Õ¹Ðì(€€€ô(€€€¥˜€¡™±½Í”¡™¥±”¤€„ô€À¤ì(€€€€€€€Ù…±¥€ô€Àì(€€€ô(€€€¥˜€ …Ù…±¥¤ì(€€€€€€€™É••}ÁÑ}µ½‘•°¡µ½‘•°¤ì(€€€€€€€É•ÑÕÉ¸9U10ì(€€€ô(€€€É•ÑÕÉ¸µ½‘•°ì)ô()Ù½¥ÁÑ}™É••}‰åÑ•Ì¡Õ¹Í¥¹•¡…È¨‰åÑ•Ì¤ì(€€€™É•”¡‰åÑ•Ì¤ì)ô()Ù½¥™É••}ÁÑ}µ½‘•°¡AQ5½‘•°¨µ½‘•°¤ì(€€€¥˜€¡µ½‘•°€ôô9U10¤ì(€€€€€€€É•ÑÕÉ¸ì(€€€ô(€€€™½È€¡¥¹ÐÀ€ô€ÀìÀ€ðµ½‘•°´ùÁ…É…µ•Ñ•É}½Õ¹ÐìÀ¬¬¤ì(€€€€€€€™É••}Á…É…µ•Ñ•È¡µ½‘•°´ùÁ…É…µ•Ñ•ÉÍmÁt¤ì(€€€ô(€€€™É•”¡µ½‘•°´ùÁ…É…µ•Ñ•ÉÌ¤ì(€€€™É•”¡µ½‘•°´ù‰±½­Ì¤ì(€€€™É•”¡µ½‘•°¤ì)ô