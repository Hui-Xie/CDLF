//
// Created by Hui Xie on 10/1/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//
#include "DeviceKernels.h"

__global__ void deviceZeroInitialize(float *pData, const long N) {
    long index = threadIdx.x + blockIdx.x * blockDim.x;// + gridIdx.x * gridDim.x * blockDim.x;
    if (index < N) pData[index] = 0;
}

