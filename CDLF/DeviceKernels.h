//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORKERNELS_H
#define CDLF_FRAMEWORK_TENSORKERNELS_H


#include "cuda_runtime.h"

//Device is an implement carrier.


__global__ void deviceInitialize(float *pData, const long N, const float value= 0);
__global__ void device2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K);

// B = A', where B has a size M*N
__global__ void device2DMatrixTranspose(float* pA, float* pB, const long M, const long N);

#endif //CDLF_FRAMEWORK_TENSORKERNELS_H
