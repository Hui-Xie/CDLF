//
// Created by Hui Xie on 10/1/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//
#include "DeviceKernels.h"

__global__ void deviceInitialize(float *pData, const long N, const float value) {
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pData[index] = value;
        index += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

