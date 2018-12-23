#ifndef RL_NONCONVEX_TENSORBLAS_H
#define RL_NONCONVEX_TENSORBLAS_H

#include <Tensor.h>


// y = A*x+ b
// y = A'*x + b
// if  y == nullptr, b = A*x +b;
// if  y == nullptr, b = A'*x +b;
void gemv(const bool ATranspose, const Tensor<float>* pA, const Tensor<float>* px, Tensor<float>* pb, Tensor<float>* py = nullptr);


#endif