#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "transformer.h"

static GPTTrainConfig config_for(int batch_size, int steps_per_epoch) {
    GPTTrainConfig config = {
        .epochs = 1,
        .batch_size = batch_size,
        .steps_per_epoch = steps_per_epoch,
        .log_every = 0,
        .warmup_steps = 0,
        .learning_rate = 0.001f,
        .weight_decay = 0.01f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
        .grad_clip = 1.0f,
        .seed = 9
    };
    return config;
}

int main(void) {
    BPETokenizer* tokenizer = bpe_train_file("train.txt", 280);
    assert(tokenizer != NULL);
    const int batches[] = {1, 2, 7, 16, 64};

    for (size_t i = 0; i < sizeof(batches) / sizeof(batches[0]); i++) {
        GPTModel* model = create_gpt_model(
            bpe_vocab_size(tokenizer),
            8,
            16,
            2,
            1,
            32,
            (unsigned int)(3 + i)
        );
        assert(model != NULL);
        GPTTrainConfig config = config_for(batches[i], 1);
        assert(gpt_train_file(model, tokenizer, "train.txt", &config));
        free_gpt_model(model);
    }

    /* Exercise epoch traversal and its partial final mini-batch. */
    int32_t* corpus = NULL;
    int token_count = bpe_encode_file(tokenizer, "train.txt", &corpus);
    assert(token_count > 0);
    int block_count = (token_count - 1) / 8;
    free(corpus);
    assert(block_count > 7 && block_count % 7 != 0);

    GPTModel* epoch_model = create_gpt_model(
        bpe_vocab_size(tokenizer), 8, 16, 2, 1, 32, 99
    );
    assert(epoch_model != NULL);
    GPTTrainConfig epoch_config = config_for(7, 0);
    assert(gpt_train_file(epoch_model, tokenizer, "train.txt", &epoch_config));
    free_gpt_model(epoch_model);

    free_bpe_tokenizer(tokenizer);
    puts("batch sizes ok");
    return 0;
}
