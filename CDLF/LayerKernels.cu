//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "LayerKernels.h"

__global__ void deviceSigmoidDerivative(float* pX, float* pdY, const int k, float* pdX, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x; //i: thread index
    while (i < N){
        float  expx = exp(pX[i]);
        pdX[i] += pdY[i]*k*expx/pow(1+expx,2);
        i += blockDim.x*gridDim.x;
    }
}

__global__ void deviceSigmoid(float* pX, float* pY, const int k, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x; //i: thread index
    while (i < N){
        float exp_x = exp(-pX[i]);
        pY[i] = k/(1+exp_x);
        i += blockDim.x*gridDim.x;
    }

}


__global__ void deviceCrossEntropyGradient(float* pX, float* pGTX, float* pdX, const float epsilon, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x; //i: thread index
    while (i < N){
        if (0 != pX[i]){
            pdX[i] -= pGTX[i]/pX[i];
        }
        else{
            pdX[i] -= pGTX[i]/epsilon;
        }
        i += blockDim.x*gridDim.x;
    }
}