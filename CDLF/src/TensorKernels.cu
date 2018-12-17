//
// Created by Hui Xie on 10/1/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//
#include "TensorKernels.h"
#include <cmath> //for pow()


//i: thread index

/*
__global__ void deviceInitialize(float *pData, const int N, const float value) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;//i: thread index
    while (i < N){
        pData[i] = value;
        i += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// B = A', where B has a size M*N
__global__ void device2DMatrixTranspose(const float* __restrict__  pA, float* pB, const int M, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int totalN  = M*N;
    while (i < totalN){
        int m = i/N;
        int n = i%N; //i = m*N+n
        pB[i] = pA[n*M+m];
        i += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
__global__ void device2DMatrixProduct(const float* __restrict__  pA, const float* __restrict__  pB, float* pC, const int M,const int N, const int K){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int totalN  = M*N;
    while (i < totalN){
        pC[i] = 0.0f;
        int m = i/N;
        int n = i%N;
        for (int k=0; k<K; ++k){
            pC[i] += pA[m*K+k]*pB[k*N+n];
        }
        i += blockDim.x*gridDim.x;  //grid-stride loop
    }
}

// C = A*d, where C has a length of N, d is a scalar
__global__ void deviceTensorMultiply(const float* __restrict__  pA, const float d, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] * d;
        i += blockDim.x*gridDim.x;
    }
}

// C = A .* B, hadamard product of A and B; A,B,C have same size
__global__ void deviceTensorHadamard(const float* __restrict__  pA, const float* __restrict__  pB, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] * pB[i];
        i += blockDim.x*gridDim.x;
    }
}

// C = A+B, where C has a length of N
__global__ void deviceTensorAdd(const float* __restrict__  pA, const float* __restrict__  pB, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] + pB[i];
        i += blockDim.x*gridDim.x;
    }
}

// C = A+d, where C has a length of N, d is a scalar
__global__ void deviceTensorAdd(const float* __restrict__  pA, const float d, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] + d;
        i += blockDim.x*gridDim.x;
    }
}

// C = A-B, where C has a length of N
__global__ void deviceTensorSubtract(const float* __restrict__  pA, const float* __restrict__  pB, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] - pB[i];
        i += blockDim.x*gridDim.x;
    }
}

// C = A-d, where C has a length of N, d is a scalar
__global__ void deviceTensorSubtract(const float* __restrict__  pA, const float d, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i] - d;
        i += blockDim.x*gridDim.x;
    }
}

// C = A/d, where C has a length of N, d is a scalar
__global__ void deviceTensorDivide(const float* __restrict__  pA, const float d, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pA[i]/d;
        i += blockDim.x*gridDim.x;
    }
}

// C = (A-d)^2, where d is a scalar, power is element-wise
__global__ void deviceTensorDiffPower(const float* __restrict__  pA, const float d, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = pow(pA[i] - d, 2);
        i += blockDim.x*gridDim.x;
    }
}


//C = ln(A) natural logarithm
__global__ void deviceTensorLn(const float* __restrict__  pA, float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = log(pA[i]);
        i += blockDim.x*gridDim.x;
    }
}

//C = exp(A) exponential
__global__ void deviceTensorExp(const float* __restrict__  pA,float* pC, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < N){
        pC[i] = exp(pA[i]);
        i += blockDim.x*gridDim.x;
    }
}

//C = flip(A)
__global__ void deviceTensorFlip(float* pA, const int N){
    int M = N/2;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < M){
        float temp = 0;
        temp = pA[i];
        pA[i] = pA[N-i-1];
        pA[N-i-1] = temp;
        i += blockDim.x*gridDim.x;
    }
}

//C is subtensor of A starting at tlIndex,with span, stride
__global__ void deviceSubTensorFromTopLeft(const float* pA,const int* pTensorDimsSpan, const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,float* pC,const int N){
    int t = threadIdx.x + blockIdx.x * blockDim.x; //t indicates thread index
    while (t < N){
        //generate index;
        extern __shared__ int pIndex[];
        int n = t;
        for (int i = 0; i <spanSize; ++i) {
            pIndex[i] = n / pSubDimsSpan[i];
            n -= pIndex[i] * pSubDimsSpan[i];
        }

        //generate offset to source data
        int AIndex = 0;
        for (int i = 0; i <spanSize; ++i) {
            AIndex += (pTlIndex[i]+pIndex[i]*stride)*pTensorDimsSpan[i];
        }
        pC[t] = pA[AIndex];

        t += blockDim.x*gridDim.x;
    }
}

__global__ void deviceSubTensorFromTopLeft(const unsigned char * pA,const int* pTensorDimsSpan, const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,float* pC,const int N){
    int t = threadIdx.x + blockIdx.x * blockDim.x; //t indicates thread index
    while (t < N){
        //generate index;
        extern __shared__ int pIndex[];
        int n = t;
        for (int i = 0; i <spanSize; ++i) {
            pIndex[i] = n / pSubDimsSpan[i];
            n -= pIndex[i] * pSubDimsSpan[i];
        }

        //generate offset to source data
        int AIndex = 0;
        for (int i = 0; i <spanSize; ++i) {
            AIndex += (pTlIndex[i]+pIndex[i]*stride)*pTensorDimsSpan[i];
        }
        pC[t] = (float) pA[AIndex];

        t += blockDim.x*gridDim.x;
    }
}

__global__ void deviceSubTensorFromTopLeft(const unsigned char * pA,const int* pTensorDimsSpan, const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,unsigned char* pC,const int N){
    int t = threadIdx.x + blockIdx.x * blockDim.x; //t indicates thread index
    while (t < N){
        //generate index;
        extern __shared__ int pIndex[];
        int n = t;
        for (int i = 0; i <spanSize; ++i) {
            pIndex[i] = n / pSubDimsSpan[i];
            n -= pIndex[i] * pSubDimsSpan[i];
        }

        //generate offset to source data
        int AIndex = 0;
        for (int i = 0; i <spanSize; ++i) {
            AIndex += (pTlIndex[i]+pIndex[i]*stride)*pTensorDimsSpan[i];
        }
        pC[t] = pA[AIndex];

        t += blockDim.x*gridDim.x;
    }
}

*/