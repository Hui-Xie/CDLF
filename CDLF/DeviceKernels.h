//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORKERNELS_H
#define CDLF_FRAMEWORK_TENSORKERNELS_H


#include "cuda_runtime.h"

//Device is an implement carrier.


__global__ void deviceInitialize(float *pData, const long N, const float value= 0);

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
__global__ void device2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K);

// B = A', where B has a size M*N
__global__ void device2DMatrixTranspose(float* pA, float* pB, const long M, const long N);

// C = A+B, where C has a length of N
__global__ void deviceTensorAdd(float* pA, float* pB, float* pC, const long N);

// C = A+d, where C has a length of N, d is scalar
__global__ void deviceTensorAdd(float* pA, const float d, float* pC, const long N);

#endif //CDLF_FRAMEWORK_TENSORKERNELS_H
