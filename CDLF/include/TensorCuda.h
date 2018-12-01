//
// Created by Hui Xie on 10/2/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#ifndef CDLF_FRAMEWORK_TENSORDEVICE_H
#define CDLF_FRAMEWORK_TENSORDEVICE_H

//cuda is a bridge between Tensor and Device



/*  old Cuda implementation.

void cudaInitialize(float* m_data, const long N, const float value=0);

// B = A', where B has a size M*N
void cuda2DMatrixTranspose(float* pA, float* pB, const long M, const long N);

// C = A*B, where A has a size of M*K, B has a size of K*N, C will has a size of M*N
void cuda2DMatrixProduct(float* pA, float* pB, float* pC, const long M,const long N, const long K);

// C = A*d, where C has a length of N, d is a scalar
void cudaTensorMultiply(float* pA, const float d, float* pC, const long N);

// C = A .* B, hadamard product of A and B; A,B,C have same size
void cudaTensorHadamard(float* pA, float* pB, float* pC, const long N);

// C = A+B, where C has a length of N
void cudaTensorAdd(float* pA, float* pB, float* pC, const long N);

// C = A+d, where C has a length of N, d is a scalar
void cudaTensorAdd(float* pA, const float d, float* pC, const long N);


// C = A-B, where C has a length of N
void cudaTensorSubtract(float* pA, float*  pB, float* pC, const long N);

// C = A-d, where C has a length of N, d is a scalar
void cudaTensorSubtract(float* pA, const float d, float* pC, const long N);

// C = A/d, where C has a length of N, d is a scalar
void cudaTensorDivide(float* pA, const float d, float* pC, const long N);

// C = (A-d)^2, where d is a scalar, power is element-wise
void cudaTensorDiffPower(float* pA, const float d, float* pC, const long N);

//C = ln(A) natural logarithm
void cudaTensorLn(float* pA, float* pC, const long N);

//C = exp(A) exponential
void cudaTensorExp(float* pA,float* pC, const long N);


//C = flip(A)
void cudaTensorFlip(float* pA, const long N);

//C is subtensor of A starting at tlIndex,with span, stride
void cudaSubTensorFromTopLeft(const float* pA,const long* pTensorDimsSpan,const long* pTlIndex, const long* pSubDimsSpan, const int spanSize, const int stride,float* pC,const long N);
void cudaSubTensorFromTopLeft(const unsigned char * pA,const long* pTensorDimsSpan,const long* pTlIndex, const long* pSubDimsSpan, const int spanSize, const int stride,float* pC,const long N);
void cudaSubTensorFromTopLeft(const unsigned char * pA,const long* pTensorDimsSpan,const long* pTlIndex, const long* pSubDimsSpan, const int spanSize, const int stride,unsigned char* pC,const long N);

*/


#endif //CDLF_FRAMEWORK_TENSORDEVICE_H
