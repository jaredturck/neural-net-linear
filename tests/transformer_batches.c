#include <assert.h>
#include <stdio.h>
#include "tokenizer.h"
#include "transformer.h"

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
        GPTTrainConfig config = {
            .epochs = 1,
            .batch_size = batches[i],
            .steps_per_epoch = 1,
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
        assert(gpt_train_file(model, tokenizer, "train.txt", &config));
        free_gpt_model(model);
    }

    free_bpe_tokenizer(tokenizer);
    puts("batch sizes ok");
    return 0;
}
