//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "LayerCuda.h"
#include "LayerKernels.h"
#include "cuda_runtime.h"

void cudaSigmoidDerivative(float* pX, float* pdY, const int k, float* pdX, const long N){
    deviceSigmoidDerivative<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pdY, k,pdX, N);
    cudaDeviceSynchronize();
}
