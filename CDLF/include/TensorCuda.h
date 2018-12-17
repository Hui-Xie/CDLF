//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORDEVICE_H
#define CDLF_FRAMEWORK_TENSORDEVICE_H

//cuda is a bridge between Tensor and Device



/*  old Cuda implementation.

void cudaInitialize(float* m_data, const int N, const float value=0);

// B = A', where B has a size M*N
void cuda2DMatrixTranspose(float* pA, float* pB, const int M, const int N);

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
void cuda2DMatrixProduct(float* pA, float* pB, float* pC, const int M,const int N, const int K);

// C = A*d, where C has a length of N, d is a scalar
void cudaTensorMultiply(float* pA, const float d, float* pC, const int N);

// C = A .* B, hadamard product of A and B; A,B,C have same size
void cudaTensorHadamard(float* pA, float* pB, float* pC, const int N);

// C = A+B, where C has a length of N
void cudaTensorAdd(float* pA, float* pB, float* pC, const int N);

// C = A+d, where C has a length of N, d is a scalar
void cudaTensorAdd(float* pA, const float d, float* pC, const int N);


// C = A-B, where C has a length of N
void cudaTensorSubtract(float* pA, float*  pB, float* pC, const int N);

// C = A-d, where C has a length of N, d is a scalar
void cudaTensorSubtract(float* pA, const float d, float* pC, const int N);

// C = A/d, where C has a length of N, d is a scalar
void cudaTensorDivide(float* pA, const float d, float* pC, const int N);

// C = (A-d)^2, where d is a scalar, power is element-wise
void cudaTensorDiffPower(float* pA, const float d, float* pC, const int N);

//C = ln(A) natural logarithm
void cudaTensorLn(float* pA, float* pC, const int N);

//C = exp(A) exponential
void cudaTensorExp(float* pA,float* pC, const int N);


//C = flip(A)
void cudaTensorFlip(float* pA, const int N);

//C is subtensor of A starting at tlIndex,with span, stride
void cudaSubTensorFromTopLeft(const float* pA,const int* pTensorDimsSpan,const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,float* pC,const int N);
void cudaSubTensorFromTopLeft(const unsigned char * pA,const int* pTensorDimsSpan,const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,float* pC,const int N);
void cudaSubTensorFromTopLeft(const unsigned char * pA,const int* pTensorDimsSpan,const int* pTlIndex, const int* pSubDimsSpan, const int spanSize, const int stride,unsigned char* pC,const int N);

*/


#endif //CDLF_FRAMEWORK_TENSORDEVICE_H
