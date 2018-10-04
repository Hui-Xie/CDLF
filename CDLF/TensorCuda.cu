//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "TensorCuda.h"
#include "DeviceKernels.h"
#include "GPUAttr.h"

void cudaInitialize(float* m_data, const long N, const float value){
    deviceInitialize<<<g_blocksPerGrid, g_maxThreadsPerBlock>>>(m_data, N, value);
    cudaDeviceSynchronize();
}

// C = A*B, where A has a size of M*K, B has a size of k*N, C will has a size of M*N
void cuda2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K){
    device2DMatrixProduct<<<g_blocksPerGrid, g_maxThreadsPerBlock>>>(pA,pB,pB, M, N, K);
    cudaDeviceSynchronize();
}