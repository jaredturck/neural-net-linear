#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"
#include "transformer.h"

static void write_repeat(const char* path, const char* text, int repeat) {
    FILE* file = fopen(path, "wb");
    assert(file != NULL);
    for (int i = 0; i < repeat; i++) {
        assert(fwrite(text, 1, strlen(text), file) == strlen(text));
    }
    assert(fclose(file) == 0);
}

static int same_file(const char* first_path, const char* second_path) {
    FILE* first = fopen(first_path, "rb");
    FILE* second = fopen(second_path, "rb");
    assert(first != NULL && second != NULL);

    int a;
    int b;
    do {
        a = fgetc(first);
        b = fgetc(second);
        if (a != b) {
            fclose(first);
            fclose(second);
            return 0;
        }
    } while (a != EOF);

    int valid = !ferror(first) && !ferror(second);
    assert(fclose(first) == 0);
    assert(fclose(second) == 0);
    return valid;
}

int main(void) {
    const char* first_path = "/tmp/nn_edges_first.txt";
    const char* second_path = "/tmp/nn_edges_second.txt";
    const char* tokenizer_path = "/tmp/nn_edges.bpe";
    const char* model_path = "/tmp/nn_edges.gpt";
    const char* resumed_a_path = "/tmp/nn_edges_resumed_a.gpt";
    const char* resumed_b_path = "/tmp/nn_edges_resumed_b.gpt";
    const char* corrupt_model_path = "/tmp/nn_edges_corrupt.gpt";

    write_repeat(first_path, "cat mammal spider arachnid salmon fish\n", 100);
    write_repeat(second_path, "alpha beta gamma delta epsilon zeta\n", 100);

    BPETokenizer* first = bpe_train_file(first_path, 260);
    BPETokenizer* second = bpe_train_file(second_path, 260);
    assert(first != NULL && second != NULL);
    assert(bpe_vocab_size(first) == bpe_vocab_size(second));
    assert(bpe_fingerprint(first) != bpe_fingerprint(second));
    assert(bpe_encode(first, NULL, 0, NULL, 0) == 0);
    assert(bpe_decode(first, NULL, 0, NULL, 0) == 0);

    assert(bpe_save(first, tokenizer_path));
    BPETokenizer* loaded_tokenizer = bpe_load(tokenizer_path);
    assert(loaded_tokenizer != NULL);
    assert(bpe_fingerprint(first) == bpe_fingerprint(loaded_tokenizer));

    GPTModel* model = create_gpt_model(260, 8, 8, 2, 1, 16, 7);
    assert(model != NULL);
    assert(gpt_bind_tokenizer(model, first));
    assert(!gpt_bind_tokenizer(model, second));

    GPTTrainConfig invalid = {
        .epochs = 1,
        .batch_size = 2,
        .steps_per_epoch = 1,
        .log_every = 0,
        .warmup_steps = 0,
        .learning_rate = NAN,
        .weight_decay = 0.01f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
        .grad_clip = 1.0f,
        .seed = 1
    };
    assert(!gpt_train_file(model, first, first_path, &invalid));

    int generated_bytes = 0;
    unsigned char* generated = gpt_generate(
        model,
        first,
        (const unsigned char*)"cat",
        3,
        0,
        0.0f,
        0,
        1,
        &generated_bytes
    );
    assert(generated != NULL);
    assert(generated_bytes == 3);
    assert(memcmp(generated, "cat", 3) == 0);
    gpt_free_bytes(generated);

    assert(gpt_generate(
        model,
        second,
        (const unsigned char*)"cat",
        3,
        1,
        0.0f,
        0,
        1,
        &generated_bytes
    ) == NULL);

    GPTTrainConfig resume_config = {
        .epochs = 1,
        .batch_size = 2,
        .steps_per_epoch = 1,
        .log_every = 0,
        .warmup_steps = 4,
        .learning_rate = 0.001f,
        .weight_decay = 0.01f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
        .grad_clip = 1.0f,
        .seed = 44
    };
    assert(gpt_train_file(model, first, first_path, &resume_config));
    assert(gpt_save(model, model_path));

    GPTModel* loaded = gpt_load(model_path);
    assert(loaded != NULL);
    assert(gpt_tokenizer_fingerprint(loaded) == bpe_fingerprint(first));
    assert(gpt_bind_tokenizer(loaded, loaded_tokenizer));
    assert(!gpt_bind_tokenizer(loaded, second));

    /* AdamW moments and optimizer step must survive checkpointing exactly. */
    assert(gpt_train_file(model, first, first_path, &resume_config));
    assert(gpt_train_file(loaded, loaded_tokenizer, first_path, &resume_config));
    assert(gpt_save(model, resumed_a_path));
    assert(gpt_save(loaded, resumed_b_path));
    assert(same_file(resumed_a_path, resumed_b_path));

    FILE* input = fopen(model_path, "rb");
    FILE* output = fopen(corrupt_model_path, "wb");
    assert(input != NULL && output != NULL);
    int ch;
    while ((ch = fgetc(input)) != EOF) {
        assert(fputc(ch, output) != EOF);
    }
    assert(fputc('x', output) != EOF);
    assert(fclose(input) == 0);
    assert(fclose(output) == 0);
    assert(gpt_load(corrupt_model_path) == NULL);

    free_gpt_model(loaded);
    free_gpt_model(model);
    free_bpe_tokenizer(loaded_tokenizer);
    free_bpe_tokenizer(first);
    free_bpe_tokenizer(second);
    puts("transformer edge cases ok");
    return 0;
}
