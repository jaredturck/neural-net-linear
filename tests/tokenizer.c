#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

int main(void) {
    const char* text = "banana bandana banana bandana\n";
    FILE* file = fopen("/tmp/nn_bpe_corpus.txt", "wb");
    assert(file != NULL);
    for (int i = 0; i < 20; i++) {
        assert(fwrite(text, 1, strlen(text), file) == strlen(text));
    }
    assert(fclose(file) == 0);

    BPETokenizer* tokenizer = bpe_train_file("/tmp/nn_bpe_corpus.txt", 280);
    assert(tokenizer != NULL);
    assert(bpe_vocab_size(tokenizer) > 256);
    assert(bpe_fingerprint(tokenizer) != 0);
    assert(bpe_encode(tokenizer, NULL, 0, NULL, 0) == 0);
    assert(bpe_decode(tokenizer, NULL, 0, NULL, 0) == 0);

    const unsigned char input[] = "banana bandana";
    int count = bpe_encode(tokenizer, input, (int)strlen((const char*)input), NULL, 0);
    assert(count > 0 && count < (int)strlen((const char*)input));
    int32_t* ids = malloc((size_t)count * sizeof(int32_t));
    assert(ids != NULL);
    assert(bpe_encode(tokenizer, input, (int)strlen((const char*)input), ids, count) == count);

    int byte_count = bpe_decode(tokenizer, ids, count, NULL, 0);
    assert(byte_count == (int)strlen((const char*)input));
    unsigned char* output = malloc((size_t)byte_count + 1);
    assert(output != NULL);
    assert(bpe_decode(tokenizer, ids, count, output, byte_count) == byte_count);
    output[byte_count] = 0;
    assert(strcmp((const char*)output, (const char*)input) == 0);

    assert(bpe_save(tokenizer, "/tmp/nn_test.bpe"));
    BPETokenizer* loaded = bpe_load("/tmp/nn_test.bpe");
    assert(loaded != NULL);
    assert(bpe_fingerprint(loaded) == bpe_fingerprint(tokenizer));

    file = fopen("/tmp/nn_test.bpe", "ab");
    assert(file != NULL);
    assert(fputc('x', file) != EOF);
    assert(fclose(file) == 0);
    assert(bpe_load("/tmp/nn_test.bpe") == NULL);

    free(output);
    free(ids);
    free_bpe_tokenizer(loaded);
    free_bpe_tokenizer(tokenizer);
    puts("tokenizer ok");
    return 0;
}
