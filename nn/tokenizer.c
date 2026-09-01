#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

typedef struct {
    int32_t left;
    int32_t right;
} MergeRule;

typedef struct {
    uint64_t* keys;
    int* counts;
    unsigned char* occupied;
    size_t capacity;
} PairTable;

struct BPETokenizer {
    int vocab_size;
    int merge_count;
    int merge_capacity;
    MergeRule* merges;
    unsigned char** token_bytes;
    int* token_lengths;
};


static int checked_array_bytes(size_t count, size_t element_size, size_t* bytes) {
    if (bytes == NULL || (element_size != 0 && count > SIZE_MAX / element_size)) {
        return 0;
    }
    *bytes = count * element_size;
    return 1;
}

static uint64_t pair_key(int32_t left, int32_t right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
}

static size_t hash_pair(uint64_t value) {
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value *= UINT64_C(0xc4ceb9fe1a85ec53);
    value ^= value >> 33;
    return (size_t)value;
}

static size_t next_power_of_two(size_t value) {
    size_t result = 1;
    while (result < value && result <= SIZE_MAX / 2) {
        result <<= 1;
    }
    return result < value ? 0 : result;
}

static int pair_table_init(PairTable* table, size_t expected_pairs) {
    if (expected_pairs > SIZE_MAX / 2) {
        return 0;
    }
    size_t requested = expected_pairs < 16 ? 16 : expected_pairs * 2;
    size_t capacity = next_power_of_two(requested);
    if (capacity == 0) {
        return 0;
    }

    table->keys = calloc(capacity, sizeof(uint64_t));
    table->counts = calloc(capacity, sizeof(int));
    table->occupied = calloc(capacity, sizeof(unsigned char));
    if (table->keys == NULL || table->counts == NULL || table->occupied == NULL) {
        free(table->keys);
        free(table->counts);
        free(table->occupied);
        memset(table, 0, sizeof(*table));
        return 0;
    }
    table->capacity = capacity;
    return 1;
}

static void pair_table_free(PairTable* table) {
    free(table->keys);
    free(table->counts);
    free(table->occupied);
    memset(table, 0, sizeof(*table));
}

static int pair_table_increment(PairTable* table, int32_t left, int32_t right) {
    uint64_t key = pair_key(left, right);
    size_t mask = table->capacity - 1;
    size_t index = hash_pair(key) & mask;

    while (table->occupied[index]) {
        if (table->keys[index] == key) {
            if (table->counts[index] == INT_MAX) {
                return 0;
            }
            table->counts[index]++;
            return 1;
        }
        index = (index + 1) & mask;
    }

    table->occupied[index] = 1;
    table->keys[index] = key;
    table->counts[index] = 1;
    return 1;
}

static int pair_table_best(const PairTable* table, int32_t* left, int32_t* right) {
    int best_count = 0;
    uint64_t best_key = 0;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->occupied[i] && table->counts[i] > best_count) {
            best_count = table->counts[i];
            best_key = table->keys[i];
        }
    }

    if (best_count < 2) {
        return 0;
    }

    *left = (int32_t)(uint32_t)(best_key >> 32);
    *right = (int32_t)(uint32_t)best_key;
    return best_count;
}

static int read_file_bytes(const char* path, unsigned char** output, int* output_length) {
    if (path == NULL || output == NULL || output_length == NULL) {
        return 0;
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    size_t capacity = 4096;
    size_t length = 0;
    unsigned char* bytes = malloc(capacity);
    if (bytes == NULL) {
        fclose(file);
        return 0;
    }

    int ch;
    while ((ch = fgetc(file)) != EOF) {
        if (length == capacity) {
            if (capacity > SIZE_MAX / 2) {
                free(bytes);
                fclose(file);
                return 0;
            }
            capacity *= 2;
            unsigned char* resized = realloc(bytes, capacity);
            if (resized == NULL) {
                free(bytes);
                fclose(file);
                return 0;
            }
            bytes = resized;
        }
        bytes[length++] = (unsigned char)ch;
    }
    fclose(file);

    if (length == 0 || length > INT_MAX) {
        free(bytes);
        return 0;
    }

    *output = bytes;
    *output_length = (int)length;
    return 1;
}

static BPETokenizer* create_tokenizer(int capacity) {
    if (capacity < 256) {
        return NULL;
    }

    BPETokenizer* tokenizer = calloc(1, sizeof(BPETokenizer));
    if (tokenizer == NULL) {
        return NULL;
    }

    tokenizer->merge_capacity = capacity - 256;
    tokenizer->merges = tokenizer->merge_capacity > 0
        ? calloc((size_t)tokenizer->merge_capacity, sizeof(MergeRule))
        : NULL;
    tokenizer->token_bytes = calloc((size_t)capacity, sizeof(unsigned char*));
    tokenizer->token_lengths = calloc((size_t)capacity, sizeof(int));

    if ((tokenizer->merge_capacity > 0 && tokenizer->merges == NULL) ||
        tokenizer->token_bytes == NULL || tokenizer->token_lengths == NULL) {
        free_bpe_tokenizer(tokenizer);
        return NULL;
    }

    for (int byte = 0; byte < 256; byte++) {
        tokenizer->token_bytes[byte] = malloc(1);
        if (tokenizer->token_bytes[byte] == NULL) {
            free_bpe_tokenizer(tokenizer);
            return NULL;
        }
        tokenizer->token_bytes[byte][0] = (unsigned char)byte;
        tokenizer->token_lengths[byte] = 1;
    }
    tokenizer->vocab_size = 256;
    return tokenizer;
}

static int append_merge(BPETokenizer* tokenizer, int32_t left, int32_t right) {
    if (tokenizer == NULL || tokenizer->merge_count >= tokenizer->merge_capacity ||
        left < 0 || right < 0 || left >= tokenizer->vocab_size || right >= tokenizer->vocab_size) {
        return 0;
    }

    int new_id = 256 + tokenizer->merge_count;
    int left_length = tokenizer->token_lengths[left];
    int right_length = tokenizer->token_lengths[right];
    if (left_length > INT_MAX - right_length) {
        return 0;
    }
    int length = left_length + right_length;
    unsigned char* bytes = malloc((size_t)length);
    if (bytes == NULL) {
        return 0;
    }

    memcpy(bytes, tokenizer->token_bytes[left], (size_t)left_length);
    memcpy(bytes + left_length, tokenizer->token_bytes[right], (size_t)right_length);

    tokenizer->merges[tokenizer->merge_count].left = left;
    tokenizer->merges[tokenizer->merge_count].right = right;
    tokenizer->token_bytes[new_id] = bytes;
    tokenizer->token_lengths[new_id] = length;
    tokenizer->merge_count++;
    tokenizer->vocab_size++;
    return 1;
}

static int merge_pair_in_place(
    int32_t* tokens,
    int token_count,
    int32_t left,
    int32_t right,
    int32_t replacement
) {
    int write = 0;
    int read = 0;

    while (read < token_count) {
        if (read + 1 < token_count && tokens[read] == left && tokens[read + 1] == right) {
            tokens[write++] = replacement;
            read += 2;
        } else {
            tokens[write++] = tokens[read++];
        }
    }
    return write;
}

BPETokenizer* bpe_train_file(const char* path, int target_vocab_size) {
    if (target_vocab_size < 256) {
        return NULL;
    }

    unsigned char* bytes = NULL;
    int byte_count = 0;
    if (!read_file_bytes(path, &bytes, &byte_count)) {
        return NULL;
    }

    BPETokenizer* tokenizer = create_tokenizer(target_vocab_size);
    size_t token_bytes = 0;
    int32_t* tokens = checked_array_bytes((size_t)byte_count, sizeof(int32_t), &token_bytes)
        ? malloc(token_bytes)
        : NULL;
    if (tokenizer == NULL || tokens == NULL) {
        free(bytes);
        free(tokens);
        free_bpe_tokenizer(tokenizer);
        return NULL;
    }

    for (int i = 0; i < byte_count; i++) {
        tokens[i] = (int32_t)bytes[i];
    }
    free(bytes);

    int token_count = byte_count;
    while (tokenizer->vocab_size < target_vocab_size && token_count > 1) {
        PairTable table = {0};
        if (!pair_table_init(&table, (size_t)token_count - 1)) {
            free(tokens);
            free_bpe_tokenizer(tokenizer);
            return NULL;
        }

        int valid = 1;
        for (int i = 0; i + 1 < token_count; i++) {
            if (!pair_table_increment(&table, tokens[i], tokens[i + 1])) {
                valid = 0;
                break;
            }
        }

        int32_t left = 0;
        int32_t right = 0;
        int count = valid ? pair_table_best(&table, &left, &right) : 0;
        pair_table_free(&table);

        if (!valid) {
            free(tokens);
            free_bpe_tokenizer(tokenizer);
            return NULL;
        }
        if (count < 2) {
            break;
        }

        int32_t replacement = tokenizer->vocab_size;
        if (!append_merge(tokenizer, left, right)) {
            free(tokens);
            free_bpe_tokenizer(tokenizer);
            return NULL;
        }
        token_count = merge_pair_in_place(tokens, token_count, left, right, replacement);
    }

    free(tokens);
    return tokenizer;
}

int bpe_encode(
    BPETokenizer* tokenizer,
    const unsigned char* input,
    int input_length,
    int32_t* output,
    int output_capacity
) {
    if (tokenizer == NULL || input_length < 0 || (input_length > 0 && input == NULL)) {
        return -1;
    }
    if (input_length == 0) {
        return 0;
    }

    size_t token_bytes = 0;
    int32_t* tokens = checked_array_bytes((size_t)input_length, sizeof(int32_t), &token_bytes)
        ? malloc(token_bytes)
        : NULL;
    if (tokens == NULL) {
        return -1;
    }
    for (int i = 0; i < input_length; i++) {
        tokens[i] = (int32_t)input[i];
    }

    int token_count = input_length;
    for (int merge = 0; merge < tokenizer->merge_count; merge++) {
        MergeRule rule = tokenizer->merges[merge];
        token_count = merge_pair_in_place(
            tokens,
            token_count,
            rule.left,
            rule.right,
            256 + merge
        );
    }

    if (output != NULL) {
        if (output_capacity < token_count) {
            free(tokens);
            return -1;
        }
        memcpy(output, tokens, (size_t)token_count * sizeof(int32_t));
    }
    free(tokens);
    return token_count;
}

int bpe_encode_file(BPETokenizer* tokenizer, const char* path, int32_t** output_tokens) {
    if (tokenizer == NULL || path == NULL || output_tokens == NULL) {
        return -1;
    }

    unsigned char* bytes = NULL;
    int byte_count = 0;
    if (!read_file_bytes(path, &bytes, &byte_count)) {
        return -1;
    }

    int token_count = bpe_encode(tokenizer, bytes, byte_count, NULL, 0);
    if (token_count <= 0) {
        free(bytes);
        return -1;
    }

    size_t token_bytes = 0;
    int32_t* tokens = checked_array_bytes((size_t)token_count, sizeof(int32_t), &token_bytes)
        ? malloc(token_bytes)
        : NULL;
    if (tokens == NULL) {
        free(bytes);
        return -1;
    }

    int encoded = bpe_encode(tokenizer, bytes, byte_count, tokens, token_count);
    free(bytes);
    if (encoded != token_count) {
        free(tokens);
        return -1;
    }

    *output_tokens = tokens;
    return token_count;
}

int bpe_decode(
    BPETokenizer* tokenizer,
    const int32_t* tokens,
    int token_count,
    unsigned char* output,
    int output_capacity
) {
    if (tokenizer == NULL || token_count < 0 || (token_count > 0 && tokens == NULL)) {
        return -1;
    }
    if (token_count == 0) {
        return 0;
    }

    size_t total = 0;
    for (int i = 0; i < token_count; i++) {
        int32_t token = tokens[i];
        if (token < 0 || token >= tokenizer->vocab_size) {
            return -1;
        }
        total += (size_t)tokenizer->token_lengths[token];
        if (total > INT_MAX) {
            return -1;
        }
    }

    if (output != NULL) {
        if (output_capacity < (int)total) {
            return -1;
        }
        size_t offset = 0;
        for (int i = 0; i < token_count; i++) {
            int32_t token = tokens[i];
            int length = tokenizer->token_lengths[token];
            memcpy(output + offset, tokenizer->token_bytes[token], (size_t)length);
            offset += (size_t)length;
        }
    }
    return (int)total;
}

int bpe_save(BPETokenizer* tokenizer, const char* path) {
    if (tokenizer == NULL || path == NULL) {
        return 0;
    }

    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }

    static const unsigned char magic[8] = {'N', 'N', 'B', 'P', 'E', '1', '\r', '\n'};
    uint32_t merge_count = (uint32_t)tokenizer->merge_count;
    int valid = fwrite(magic, 1, sizeof(magic), file) == sizeof(magic) &&
        fwrite(&merge_count, sizeof(merge_count), 1, file) == 1;

    for (int i = 0; valid && i < tokenizer->merge_count; i++) {
        int32_t pair[2] = {tokenizer->merges[i].left, tokenizer->merges[i].right};
        valid = fwrite(pair, sizeof(int32_t), 2, file) == 2;
    }

    if (fclose(file) != 0) {
        valid = 0;
    }
    return valid;
}

BPETokenizer* bpe_load(const char* path) {
    if (path == NULL) {
        return NULL;
    }

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }

    static const unsigned char expected[8] = {'N', 'N', 'B', 'P', 'E', '1', '\r', '\n'};
    unsigned char magic[8];
    uint32_t merge_count = 0;
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        memcmp(magic, expected, sizeof(magic)) != 0 ||
        fread(&merge_count, sizeof(merge_count), 1, file) != 1 ||
        merge_count > (uint32_t)(INT_MAX - 256)) {
        fclose(file);
        return NULL;
    }

    BPETokenizer* tokenizer = create_tokenizer(256 + (int)merge_count);
    if (tokenizer == NULL) {
        fclose(file);
        return NULL;
    }

    for (uint32_t i = 0; i < merge_count; i++) {
        int32_t pair[2];
        if (fread(pair, sizeof(int32_t), 2, file) != 2 ||
            !append_merge(tokenizer, pair[0], pair[1])) {
            fclose(file);
            free_bpe_tokenizer(tokenizer);
            return NULL;
        }
    }

    if (fgetc(file) != EOF || ferror(file) || fclose(file) != 0) {
        free_bpe_tokenizer(tokenizer);
        return NULL;
    }
    return tokenizer;
}

int bpe_vocab_size(BPETokenizer* tokenizer) {
    return tokenizer == NULL ? 0 : tokenizer->vocab_size;
}

int bpe_merge_count(BPETokenizer* tokenizer) {
    return tokenizer == NULL ? 0 : tokenizer->merge_count;
}

uint64_t bpe_fingerprint(BPETokenizer* tokenizer) {
    if (tokenizer == NULL) {
        return 0;
    }

    /* FNV-1a over the tokenizer format version and ordered merge rules. */
    uint64_t hash = UINT64_C(1469598103934665603);
    const uint64_t prime = UINT64_C(1099511628211);
    const unsigned char version[] = {'N', 'N', 'B', 'P', 'E', '1'};
    for (size_t i = 0; i < sizeof(version); i++) {
        hash ^= version[i];
        hash *= prime;
    }

    uint32_t count = (uint32_t)tokenizer->merge_count;
    for (int byte = 0; byte < 4; byte++) {
        hash ^= (unsigned char)(count >> (byte * 8));
        hash *= prime;
    }

    for (int i = 0; i < tokenizer->merge_count; i++) {
        uint32_t values[2] = {
            (uint32_t)tokenizer->merges[i].left,
            (uint32_t)tokenizer->merges[i].right
        };
        for (int value = 0; value < 2; value++) {
            for (int byte = 0; byte < 4; byte++) {
                hash ^= (unsigned char)(values[value] >> (byte * 8));
                hash *= prime;
            }
        }
    }

    /* Reserve zero for an unbound model. */
    return hash == 0 ? UINT64_C(1) : hash;
}

void free_bpe_tokenizer(BPETokenizer* tokenizer) {
    if (tokenizer == NULL) {
        return;
    }

    if (tokenizer->token_bytes != NULL) {
        int capacity = 256 + tokenizer->merge_capacity;
        for (int i = 0; i < capacity; i++) {
            free(tokenizer->token_bytes[i]);
        }
    }
    free(tokenizer->token_bytes);
    free(tokenizer->token_lengths);
    free(tokenizer->merges);
    free(tokenizer);
}
