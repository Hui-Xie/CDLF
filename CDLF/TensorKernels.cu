//
// Created by Hui Xie on 10/1/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//
#include "TensorKernels.h"
#include <cmath> //for pow()


//i: thread index

__global__ void deviceInitialize(float *pData, const long N, const float value) {
    long i = threadIdx.x + blockIdx.x * blockDim.x;//i: thread index
    while (i < N){
        pData[i] = value;
        i += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// B = A', where B has a size M*N
__global__ void device2DMatrixTranspose(float* pA, float* pB, const long M, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long totalN  = M*N;
    while (i < totalN){
        long m = i/N;
        long n = i%N; //i = m*N+n
        pB[i] = pA[n*M+m];
        i += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
__global__ void device2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long totalN  = M*N;
    while (i < totalN){
        pC[i] = 0.0f;
        long m = i/N;
        long n = i%N;
        for (long k=0; k<K; ++k){
            pC[i] += pA[m*K+k]*pB[k*N+n];
        }
        i += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// C = A*d, where C has a length of N, d is a scalar
__global__ void deviceTensorMultiply(float* pA, const float d, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] * d;
        i += blockDim.x*gridDim.x;
    }
}

// C = A .* B, hadamard product of A and B; A,B,C have same size
__global__ void deviceTensorHadamard(float* pA, float* pB, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] * pB[i];
        i += blockDim.x*gridDim.x;
    }
}

// C = A+B, where C has a length of N
__global__ void deviceTensorAdd(float* pA, float* pB, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] + pB[i];
        i += blockDim.x*gridDim.x;
    }
}

// C = A+d, where C has a length of N, d is a scalar
__global__ void deviceTensorAdd(float* pA, const float d, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] + d;
        i += blockDim.x*gridDim.x;
    }
}

// C = A-B, where C has a length of N
__global__ void deviceTensorSubtract(float* pA, float* pB, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] - pB[i];
        i += blockDim.x*gridDim.x;
    }
}

// C = A-d, where C has a length of N, d is a scalar
__global__ void deviceTensorSubtract(float* pA, const float d, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] - d;
        i += blockDim.x*gridDim.x;
    }
}

// C = A/d, where C has a length of N, d is a scalar
__global__ void deviceTensorDivide(float* pA, const float d, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i]/d;
        i += blockDim.x*gridDim.x;
    }
}

// C = (A-d)^2, where d is a scalar, power is element-wise
__global__ void deviceTensorDiffPower(float* pA, const float d, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pow(pA[i] - d, 2);
        i += blockDim.x*gridDim.x;
    }
}


//C = ln(A) natural logarithm
__global__ void deviceTensorLn(float* pA, float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = log(pA[i]);
        i += blockDim.x*gridDim.x;
    }
}

//C = exp(A) exponential
__global__ void deviceTensorExp(float* pA,float* pC, const long N){
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = exp(pA[i]);
        i += blockDim.x*gridDim.x;
    }
}

//C = flip(A)
__global__ void deviceTensorFlip(float* pA, const long N){
    long M = N/2;
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < M){
        float temp = 0;
        temp = pA[i];
        pA[i] = pA[N-i-1];
        pA[N-i-1] = temp;
        i += blockDim.x*gridDim.x;
    }
}