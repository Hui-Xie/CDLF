#ifndef RL_NONCONVEX_TENSORBLAS_H
#define RL_NONCONVEX_TENSORBLAS_H

#include <Tensor.h>


// y = A*x+ b
// y = A'*x + b
// if  y == nullptr, b = A*x +b;
// if  y == nullptr, b = A'*x +b;
void gemv(const bool ATranspose, const Tensor<float>* pA, const Tensor<float>* px, Tensor<float>* pb, Tensor<float>* py = nullptr);

// y = ax+y
void axpy(const float a, const Tensor<float>* px, Tensor<float>* py);

// C = a*A*B+ b*C
void gemm(const float a, const bool ATranpose, const Tensor<float>* pA, const bool BTranspose, const Tensor<float>* pB, const float b, Tensor<float>* pC);

// C = a*A + b*B
void matAdd(const float a, const Tensor<float>* pA, const float b, const Tensor<float>* pB, Tensor<float>* pC);
#endif