#include <stdint.h>
#include <stdlib.h>
#include "samplers.h"

struct Sampler {
    Dataset* dataset;
    SamplerKind kind;
    SamplerStrategy strategy;
    int sequence_length;
    int count;
    int index_count;
    int cursor;
    int* indices;
    uint32_t state;
};

static uint32_t next_random(Sampler* sampler) {
    sampler->state = sampler->state * 1664525u + 1013904223u;
    return sampler->state;
}

static int random_bounded(Sampler* sampler, int limit) {
    if (limit <= 1) {
        return 0;
    }
    return (int)(next_random(sampler) % (uint32_t)limit);
}

static void shuffle_indices(Sampler* sampler) {
    for (int i=sampler->index_count - 1; i>0; i--) {
        int j = random_bounded(sampler, i + 1);
        int temp = sampler->indices[i];
        sampler->indices[i] = sampler->indices[j];
        sampler->indices[j] = temp;
    }
}

static Sampler* allocate_sampler(Dataset* dataset, SamplerKind kind, SamplerStrategy strategy, unsigned int seed) {
    Sampler* sampler = calloc(1, sizeof(Sampler));
    if (sampler == NULL) {
        return NULL;
    }
    sampler->dataset = dataset;
    sampler->kind = kind;
    sampler->strategy = strategy;
    sampler->state = seed == 0 ? 1u : (uint32_t)seed;
    return sampler;
}

Sampler* create_row_sampler(Dataset* dataset, SamplerStrategy strategy, unsigned int seed) {
    if (dataset == NULL || dataset_kind(dataset) != DATASET_TABLE || dataset_size(dataset) <= 0) {
        return NULL;
    }

    Sampler* sampler = allocate_sampler(dataset, SAMPLER_ROWS, strategy, seed);
    if (sampler == NULL) {
        return NULL;
    }
    sampler->count = dataset_size(dataset);
    sampler->index_count = sampler->count;

    if (strategy != SAMPLER_RANDOM) {
        sampler->indices = malloc((size_t)sampler->count * sizeof(int));
        if (sampler->indices == NULL) {
            free(sampler);
            return NULL;
        }
        for (int i=0; i<sampler->count; i++) {
            sampler->indices[i] = i;
        }
    }

    sampler_reset(sampler);
    return sampler;
}

Sampler* create_token_sampler(
    Dataset* dataset,
    int sequence_length,
    SamplerStrategy strategy,
    int samples,
    unsigned int seed
) {
    if (dataset == NULL || dataset_kind(dataset) != DATASET_TEXT || sequence_length <= 0 ||
        dataset_token_count(dataset) <= sequence_length) {
        return NULL;
    }

    Sampler* sampler = allocate_sampler(dataset, SAMPLER_TOKENS, strategy, seed);
    if (sampler == NULL) {
        return NULL;
    }
    sampler->sequence_length = sequence_length;

    int block_count = (dataset_token_count(dataset) - 1) / sequence_length;
    if (block_count <= 0) {
        free(sampler);
        return NULL;
    }

    if (strategy == SAMPLER_RANDOM) {
        sampler->count = samples > 0 ? samples : block_count;
    } else {
        sampler->count = samples > 0 && samples < block_count ? samples : block_count;
        sampler->index_count = block_count;
        sampler->indices = malloc((size_t)block_count * sizeof(int));
        if (sampler->indices == NULL) {
            free(sampler);
            return NULL;
        }
        for (int i=0; i<block_count; i++) {
            sampler->indices[i] = i * sequence_length;
        }
    }

    sampler_reset(sampler);
    return sampler;
}

void sampler_reset(Sampler* sampler) {
    if (sampler == NULL) {
        return;
    }
    sampler->cursor = 0;
    if (sampler->strategy == SAMPLER_SHUFFLED && sampler->indices != NULL) {
        shuffle_indices(sampler);
    }
}

int sampler_next(Sampler* sampler, int* index_or_offset) {
    if (sampler == NULL || index_or_offset == NULL || sampler->cursor >= sampler->count) {
        return 0;
    }

    if (sampler->strategy == SAMPLER_RANDOM) {
        if (sampler->kind == SAMPLER_ROWS) {
            *index_or_offset = random_bounded(sampler, dataset_size(sampler->dataset));
        } else {
            int max_start = dataset_token_count(sampler->dataset) - sampler->sequence_length;
            *index_or_offset = random_bounded(sampler, max_start);
        }
    } else {
        *index_or_offset = sampler->indices[sampler->cursor];
    }

    sampler->cursor++;
    return 1;
}

int sampler_count(Sampler* sampler) {
    return sampler == NULL ? 0 : sampler->count;
}

int sampler_sequence_length(Sampler* sampler) {
    return sampler == NULL ? 0 : sampler->sequence_length;
}

SamplerKind sampler_kind(Sampler* sampler) {
    return sampler == NULL ? SAMPLER_ROWS : sampler->kind;
}

void free_sampler(Sampler* sampler) {
    if (sampler == NULL) {
        return;
    }
    free(sampler->indices);
    free(sampler);
}
