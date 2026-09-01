#include <stdlib.h>
#include <string.h>
#include "tensor.h"

size_t tensor_element_size(TensorDType dtype) {
    switch (dtype) {
        case TENSOR_F32:
            return sizeof(float);
        case TENSOR_I32:
            return sizeof(int32_t);
    }
    return 0;
}

Tensor* tensor_create(TensorDType dtype, int ndim, const size_t* shape) {
    if (ndim <= 0 || ndim > 4 || shape == NULL) {
        return NULL;
    }

    size_t element_size = tensor_element_size(dtype);
    if (element_size == 0) {
        return NULL;
    }

    Tensor* tensor = calloc(1, sizeof(Tensor));
    if (tensor == NULL) {
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->size = 1;

    for (int i = 0; i < ndim; i++) {
        if (shape[i] == 0 || tensor->size > SIZE_MAX / shape[i]) {
            free(tensor);
            return NULL;
        }
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    if (tensor->size > SIZE_MAX / element_size) {
        free(tensor);
        return NULL;
    }

    tensor->data = calloc(tensor->size, element_size);
    if (tensor->data == NULL) {
        free(tensor);
        return NULL;
    }

    return tensor;
}

void tensor_zero(Tensor* tensor) {
    if (tensor == NULL || tensor->data == NULL) {
        return;
    }
    memset(tensor->data, 0, tensor->size * tensor_element_size(tensor->dtype));
}

void free_tensor(Tensor* tensor) {
    if (tensor == NULL) {
        return;
    }
    free(tensor->data);
    free(tensor);
}

void matmul_forward(
    const float* input,
    const float* weights,
    float* output,
    size_t rows,
    int input_size,
    int output_size
) {
    if (input == NULL || weights == NULL || output == NULL ||
        rows == 0 || input_size <= 0 || output_size <= 0) {
        return;
    }

    for (size_t row = 0; row < rows; row++) {
        const float* restrict x = input + row * (size_t)input_size;
        float* restrict y = output + row * (size_t)output_size;

        for (int output_index = 0; output_index < output_size; output_index++) {
            const float* restrict weight_row =
                weights + (size_t)output_index * (size_t)input_size;
            float total = 0.0f;
            for (int input_index = 0; input_index < input_size; input_index++) {
                total += x[input_index] * weight_row[input_index];
            }
            y[output_index] = total;
        }
    }
}

void matmul_backward_accumulate(
    const float* input,
    const float* weights,
    const float* output_gradients,
    float* input_gradients,
    float* weight_gradients,
    size_t rows,
    int input_size,
    int output_size
) {
    if (input == NULL || weights == NULL || output_gradients == NULL ||
        input_gradients == NULL || weight_gradients == NULL ||
        rows == 0 || input_size <= 0 || output_size <= 0) {
        return;
    }

    for (size_t row = 0; row < rows; row++) {
        const float* restrict x = input + row * (size_t)input_size;
        const float* restrict dy = output_gradients + row * (size_t)output_size;
        float* restrict dx = input_gradients + row * (size_t)input_size;

        for (int output_index = 0; output_index < output_size; output_index++) {
            const float gradient = dy[output_index];
            const size_t weight_offset = (size_t)output_index * (size_t)input_size;
            const float* restrict weight_row = weights + weight_offset;
            float* restrict gradient_row = weight_gradients + weight_offset;

            for (int input_index = 0; input_index < input_size; input_index++) {
                gradient_row[input_index] += gradient * x[input_index];
                dx[input_index] += gradient * weight_row[input_index];
            }
        }
    }
}
