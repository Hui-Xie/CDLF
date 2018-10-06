//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "TensorCuda.h"
#include "DeviceKernels.h"
#include "GPUAttr.h"

void cudaInitialize(float* m_data, const long N, const float value){
    deviceInitialize<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(m_data, N, value);
    cudaDeviceSynchronize();
}

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
void cuda2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K){
    device2DMatrixProduct<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA,pB,pC, M, N, K);
    cudaDeviceSynchronize();
}

// B = A', where B has a size M*N
void cuda2DMatrixTranspose(float* pA, float* pB, const long M, const long N){
    device2DMatrixTranspose<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA,pB,M, N);
    cudaDeviceSynchronize();

}

// C = A+B, where C has a length of N
void cudaTensorAdd(float* pA, float* pB, float* pC, const long N){
    deviceTensorAdd<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pB, pC, N);
    cudaDeviceSynchronize();
}

// C = A+d, where C has a length of N, d is scalar
void cudaTensorAdd(float* pA, const float d, float* pC, const long N){
    deviceTensorAdd<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

// C = A-B, where C has a length of N
void cudaTensorSubtraction(float* pA, float* pB, float* pC, const long N){
    deviceTensorSubtraction<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pB, pC, N);
    cudaDeviceSynchronize();
}

// C = A-d, where C has a length of N, d is scalar
void cudaTensorSubtraction(float* pA, const float d, float* pC, const long N){
    deviceTensorSubtraction<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

// C = A/d, where C has a length of N, d is scalar
void cudaTensorDivide(float* pA, const float d, float* pC, const long N){
    deviceTensorDivide<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}