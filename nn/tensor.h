#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TENSOR_F32,
    TENSOR_I32
} TensorDType;

typedef struct {
    void* data;
    TensorDType dtype;
    int ndim;
    size_t shape[4];
    size_t size;
} Tensor;

Tensor* tensor_create(TensorDType dtype, int ndim, const size_t* shape);
void tensor_zero(Tensor* tensor);
void free_tensor(Tensor* tensor);
size_t tensor_element_size(TensorDType dtype);

void matmul_forward(
    const float* input,
    const float* weights,
    float* output,
    size_t rows,
    int input_size,
    int output_size
);

void matmul_backward_accumulate(
    const float* input,
    const float* weights,
    const float* output_gradients,
    float* input_gradients,
    float* weight_gradients,
    size_t rows,
    int input_size,
    int output_size
);

#endif
