//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORKERNELS_H
#define CDLF_FRAMEWORK_TENSORKERNELS_H


#include "cuda_runtime.h"

//Device is an implement carrier.


__global__ void deviceInitialize(float *pData, const int N, const float value= 0);

// B = A', where B has a size M*N
__global__ void device2DMatrixTranspose(const float* __restrict__  pA, float* pB, const int M, const int N);

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
__global__ void device2DMatrixProduct(const float* __restrict__ pA, const float* __restrict__ pB, float* pC, const int M,const int N, const int K);

// C = A*d, where C has a length of N, d is a scalar
__global__ void deviceTensorMultiply(const float* __restrict__ pA, const float d, float* pC, const int N);

// C = A .* B, hadamard product of A and B; A,B,C have same size
__global__ void deviceTensorHadamard(const float* __restrict__ pA, const float* __restrict__ pB, float* pC, const int N);

// C = A+B, where C has a length of N
__global__ void deviceTensorAdd(const float* __restrict__ pA, const float* __restrict__ pB, float* pC, const int N);

// C = A+d, where C has a length of N, d is a scalar
__global__ void deviceTensorAdd(const float* __restrict__ pA, const float d, float* pC, const int N);

// C = A-B, where C has a length of N
__global__ void deviceTensorSubtract(const float* __restrict__ pA, const float* __restrict__ pB, float* pC, const int N);

// C = A-d, where C has a length of N, d is a scalar
__global__ void deviceTensorSubtract(const float* __restrict__ pA, const float d, float* pC, const int N);

// C = A/d, where C has a length of N, d is a scalar
__global__ void deviceTensorDivide(const float* __restrict__ pA, const float d, float* pC, const int N);

// C = (A-d)^2, where d is a scalar, power is element-wise
__global__ void deviceTensorDiffPower(const float* __restrict__ pA, const float d, float* pC, const int N);

//C = ln(A) natural logarithm
__global__ void deviceTensorLn(const float* __restrict__ pA, float* pC, const int N);

//C = exp(A) exponential
__global__ void deviceTensorExp(const float* __restrict__ pA,float* pC, const int N);

//C = flip(A)
__global__ void deviceTensorFlip(float* pA, const int N);


//C is subtensor of A starting at tlIndex,with span, stride
__global__ void deviceSubTensorFromTopLeft(const float* pA,const int* pTensorDimsSpan, const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,float* pC,const int N);
__global__ void deviceSubTensorFromTopLeft(const unsigned char* pA,const int* pTensorDimsSpan, const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,float* pC,const int N);
__global__ void deviceSubTensorFromTopLeft(const unsigned char* pA,const int* pTensorDimsSpan, const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,unsigned char* pC,const int N);
#endif //CDLF_FRAMEWORK_TENSORKERNELS_H
