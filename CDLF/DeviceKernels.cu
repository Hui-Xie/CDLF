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

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
__global__ void device2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    long totalN  = M*N;
    while (index < totalN){
        pC[index] = 0.0f;
        long m = index/N;
        long n = index%N;
        for (long i=0; i<K; ++i){
            pC[index] += pA[m*K+i]*pB[i*N+n];
        }
        index += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// C = A*d, where C has a length of N, d is a scalar
__global__ void deviceTensorMultiply(float* pA, const float d, float* pC, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pC[index] = pA[index] * d;
        index += blockDim.x*gridDim.x;
    }
}

// B = A', where B has a size M*N
__global__ void device2DMatrixTranspose(float* pA, float* pB, const long M, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    long totalN  = M*N;
    while (index < totalN){
        long m = index/N;
        long n = index%N; //index = m*N+n
        pB[index] = pA[n*M+m];
        index += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// C = A+B, where C has a length of N
__global__ void deviceTensorAdd(float* pA, float* pB, float* pC, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pC[index] = pA[index] + pB[index];
        index += blockDim.x*gridDim.x;
    }
}

// C = A+d, where C has a length of N, d is a scalar
__global__ void deviceTensorAdd(float* pA, const float d, float* pC, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pC[index] = pA[index] + d;
        index += blockDim.x*gridDim.x;
    }
}

// C = A-B, where C has a length of N
__global__ void deviceTensorSubtraction(float* pA, float* pB, float* pC, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pC[index] = pA[index] - pB[index];
        index += blockDim.x*gridDim.x;
    }
}

// C = A-d, where C has a length of N, d is a scalar
__global__ void deviceTensorSubtraction(float* pA, const float d, float* pC, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pC[index] = pA[index] - d;
        index += blockDim.x*gridDim.x;
    }
}

// C = A/d, where C has a length of N, d is a scalar
__global__ void deviceTensorDivide(float* pA, const float d, float* pC, const long N){
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        pC[index] = pA[index]/d;
        index += blockDim.x*gridDim.x;
    }
}