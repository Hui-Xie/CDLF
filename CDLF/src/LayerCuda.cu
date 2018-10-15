//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "LayerCuda.h"
#include "LayerKernels.h"
#include "cuda_runtime.h"
#include "GPUAttr.h"

void cudaSigmoidDerivative(const float* __restrict__  pX, const float* __restrict__  pdY, float* pdX, const int k, const long N){
    deviceSigmoidDerivative<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pdY, pdX, k, N);
    cudaDeviceSynchronize();
}

void cudaSigmoid(const float* __restrict__  pX, float* pY, const int k, const long N){
    deviceSigmoid<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pY, k,N);
    cudaDeviceSynchronize();
}

void cudaCrossEntropyGradient(const float* __restrict__  pX, const float* __restrict__  pGTX, float* pdX, const float epsilon, const long N){
    deviceCrossEntropyGradient<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pGTX, pdX, epsilon,N);
    cudaDeviceSynchronize();
}

//C = A where A and C has different value type
void cudaElementCopy(const unsigned char* __restrict__ pA,float* pC, const long N){
    deviceElementCopy<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pC, N);
    cudaDeviceSynchronize();
}

//C = A if A>=0; C =0 else
void cudaRelu(const float* __restrict__  pA,float* pC, const long N){
    deviceRelu<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pC, N);
    cudaDeviceSynchronize();
}

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
void cudaReluDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const long N){
    deviceReluDerivative<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pdY, pdX, N);
    cudaDeviceSynchronize();
}

void cudaSoftmax(const float* __restrict__  pX, float* pY, const int nSoftmax, const long N){
    deviceSoftmax<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pY, nSoftmax, N);
    cudaDeviceSynchronize();
}

void cudaSoftmaxDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int nSoftmax, const long N){
    deviceSoftmaxDerivative<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pX, pdY, pdX, nSoftmax, N);
    cudaDeviceSynchronize();
}