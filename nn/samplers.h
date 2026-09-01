#ifndef SAMPLERS_H
#define SAMPLERS_H

#include "datasets.h"

typedef enum {
    SAMPLER_SEQUENTIAL,
    SAMPLER_SHUFFLED,
    SAMPLER_RANDOM
} SamplerStrategy;

typedef enum {
    SAMPLER_ROWS,
    SAMPLER_TOKENS
} SamplerKind;

typedef struct Sampler Sampler;

Sampler* create_row_sampler(Dataset* dataset, SamplerStrategy strategy, unsigned int seed);
Sampler* create_token_sampler(
    Dataset* dataset,
    int sequence_length,
    SamplerStrategy strategy,
    int samples,
    unsigned int seed
);
void sampler_reset(Sampler* sampler);
int sampler_next(Sampler* sampler, int* index_or_offset);
int sampler_count(Sampler* sampler);
int sampler_sequence_length(Sampler* sampler);
SamplerKind sampler_kind(Sampler* sampler);
void free_sampler(Sampler* sampler);

#endif
