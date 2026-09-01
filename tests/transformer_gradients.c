#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "transformer_internal.h"

static void check_one(
    GPTModel* model,
    GPTWorkspace* workspace,
    const int32_t* input,
    const int32_t* target,
    int parameter_index,
    size_t element,
    float tolerance
) {
    float loss = train_batch(model, input, target, workspace);
    assert(isfinite(loss));
    Parameter* parameter = model->parameters[parameter_index];
    float analytic = parameter->grad[element];
    float original = parameter->data[element];
    const float epsilon = 1e-3f;

    parameter->data[element] = original + epsilon;
    float plus = train_batch(model, input, target, workspace);
    parameter->data[element] = original - epsilon;
    float minus = train_batch(model, input, target, workspace);
    parameter->data[element] = original;

    float numeric = (plus - minus) / (2.0f * epsilon);
    float denominator = fmaxf(1e-3f, fabsf(analytic) + fabsf(numeric));
    float relative_error = fabsf(analytic - numeric) / denominator;
    printf(
        "parameter=%d element=%zu analytic=%g numeric=%g relative=%g\n",
        parameter_index,
        element,
        analytic,
        numeric,
        relative_error
    );
    assert(relative_error < tolerance || fabsf(analytic - numeric) < 2e-3f);
}

static void check_single_block(void) {
    GPTModel* model = create_gpt_model(16, 3, 4, 1, 1, 8, 123);
    assert(model != NULL);
    GPTWorkspace* workspace = create_workspace(model, 2, 3);
    assert(workspace != NULL);
    const int32_t input[6] = {1, 2, 3, 4, 5, 6};
    const int32_t target[6] = {2, 3, 4, 5, 6, 7};

    /* embedding, RMS1, QKV, attention out, RMS2, gate, value, FF out, final RMS */
    check_one(model, workspace, input, target, 0, 5, 0.08f);
    check_one(model, workspace, input, target, 1, 1, 0.08f);
    check_one(model, workspace, input, target, 2, 7, 0.12f);
    check_one(model, workspace, input, target, 3, 5, 0.12f);
    check_one(model, workspace, input, target, 4, 2, 0.08f);
    check_one(model, workspace, input, target, 5, 9, 0.12f);
    check_one(model, workspace, input, target, 6, 11, 0.12f);
    check_one(model, workspace, input, target, 7, 13, 0.12f);
    check_one(model, workspace, input, target, 8, 3, 0.08f);

    free_workspace(model, workspace);
    free_gpt_model(model);
}

static void check_stacked_blocks(void) {
    GPTModel* model = create_gpt_model(16, 3, 4, 1, 2, 8, 123);
    assert(model != NULL);
    GPTWorkspace* workspace = create_workspace(model, 2, 3);
    assert(workspace != NULL);
    const int32_t input[6] = {1, 2, 3, 4, 5, 6};
    const int32_t target[6] = {2, 3, 4, 5, 6, 7};

    /* Check both blocks plus the final norm to catch cross-block residual bugs. */
    check_one(model, workspace, input, target, 2, 7, 0.15f);
    check_one(model, workspace, input, target, 7, 13, 0.15f);
    check_one(model, workspace, input, target, 9, 7, 0.15f);
    check_one(model, workspace, input, target, 14, 13, 0.15f);
    check_one(model, workspace, input, target, 15, 3, 0.08f);

    free_workspace(model, workspace);
    free_gpt_model(model);
}

int main(void) {
    check_single_block();
    check_stacked_blocks();
    puts("gradient checks ok");
    return 0;
}
