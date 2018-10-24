//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "TensorCuda.h"
#include "TensorKernels.h"
#include "GPUAttr.h"
#include <iostream>
using namespace std;



void cudaInitialize(float* m_data, const long N, const float value){
    deviceInitialize<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(m_data, N, value);
    cudaDeviceSynchronize();
}

// B = A', where B has a size M*N
void cuda2DMatrixTranspose(float* pA, float* pB, const long M, const long N){
    device2DMatrixTranspose<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA,pB,M, N);
    cudaDeviceSynchronize();

}

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
void cuda2DMatrixProduct(float* pA, float*  pB, float* pC, const long M,const long N, const long K){
    device2DMatrixProduct<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA,pB,pC, M, N, K);
    cudaDeviceSynchronize();
}

// C = A*d, where C has a length of N, d is a scalar
void cudaTensorMultiply(float*  pA, const float d, float* pC, const long N){
    deviceTensorMultiply<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

// C = A .* B, hadamard product of A and B; A,B,C have same size
void cudaTensorHadamard(float*  pA, float*  pB, float* pC, const long N){
    deviceTensorHadamard<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA,pB,pC, N);
    cudaDeviceSynchronize();
}

// C = A+B, where C has a length of N
void cudaTensorAdd(float* pA, float*  pB, float* pC, const long N){
    deviceTensorAdd<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pB, pC, N);
    cudaDeviceSynchronize();
}

// C = A+d, where C has a length of N, d is a scalar
void cudaTensorAdd(float*  pA, const float d, float* pC, const long N){
    deviceTensorAdd<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

// C = A-B, where C has a length of N
void cudaTensorSubtract(float*  pA, float*  pB, float* pC, const long N){
    deviceTensorSubtract<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pB, pC, N);
    cudaDeviceSynchronize();
}

// C = A-d, where C has a length of N, d is a scalar
void cudaTensorSubtract(float*  pA, const float d, float* pC, const long N){
    deviceTensorSubtract<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

// C = A/d, where C has a length of N, d is a scalar
void cudaTensorDivide(float*  pA, const float d, float* pC, const long N){
    deviceTensorDivide<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

// C = (A-d)^2, where d is a scalar, power is element-wise
void cudaTensorDiffPower(float*  pA, const float d, float* pC, const long N){
    deviceTensorDiffPower<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, d, pC, N);
    cudaDeviceSynchronize();
}

//C = ln(A) natural logarithm
void cudaTensorLn(float*  pA, float* pC, const long N){
    deviceTensorLn<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pC, N);
    cudaDeviceSynchronize();
}


//C = exp(A) exponential
void cudaTensorExp(float*  pA,float* pC, const long N){
    deviceTensorExp<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pC, N);
    cudaDeviceSynchronize();
}

//C = flip(A)
void cudaTensorFlip(float* pA, const long N){
    deviceTensorFlip<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, N);
    cudaDeviceSynchronize();
}

//C is subtensor of A starting at tlIndex,with span, stride
void cudaSubTensorFromTopLeft(const float* pA,const long* pTensorDimsSpan, const long* pTlIndex, const long* pSubDimsSpan, const int Ns, const int stride,float* pC,const long N){
    deviceSubTensorFromTopLeft<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pTensorDimsSpan, pTlIndex, pSubDimsSpan, Ns, stride, pC,N);
    cudaDeviceSynchronize();
}

void cudaSubTensorFromTopLeft(const unsigned char * pA,const long* pTensorDimsSpan, const long* pTlIndex, const long* pSubDimsSpan, const int Ns, const int stride,float* pC,const long N){
    deviceSubTensorFromTopLeft<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pTensorDimsSpan, pTlIndex, pSubDimsSpan, Ns, stride, pC,N);
    cudaDeviceSynchronize();
}

void cudaSubTensorFromTopLeft(const unsigned char * pA,const long* pTensorDimsSpan, const long* pTlIndex, const long* pSubDimsSpan, const int Ns, const int stride,unsigned char* pC,const long N){
    deviceSubTensorFromTopLeft<<<GPUAttr::m_blocksPerGrid, GPUAttr::m_maxThreadsPerBlock>>>(pA, pTensorDimsSpan, pTlIndex, pSubDimsSpan, Ns, stride, pC,N);
    cudaDeviceSynchronize();
}