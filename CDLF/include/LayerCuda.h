//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERCUDA_H
#define CDLF_FRAMEWORK_LAYERCUDA_H

void cudaSigmoidDerivative(const float* __restrict__  pX, const float* __restrict__  pdY, float* pdX, const int k, const int N);

void cudaSigmoid(const float* __restrict__  pX, float* pY, const int k, const int N);


void cudaCrossEntropyGradient(const float* __restrict__  pX, const float* __restrict__  pGTX, float* pdX, const float epsilon, const int N);

//C = A where A and C has different value type
void cudaElementCopy(const unsigned char* __restrict__ pA,float* pC, const int N);

//C = A if A>=0; C =0 else
void cudaRelu(const float* __restrict__  pA,float* pC, const int N);

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
void cudaReluDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int N);

void cudaSoftmax(const float* __restrict__  pX, float* pY, const int nSoftmax, const int N);

void cudaSoftmaxDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int nSoftmax, const int N);

//C = A*F in convolution
void cudaConvLayerForward(const float* pA, const int* pADimsSpan, const float* pF, const int* pFDimsSpan, const int filterSize, const int NFilter, const int stride,
                         float* pC, const int* pCDimsSpan, const int* pNonZeroIndex, const int CDimsSize, const int N);

#endif //CDLF_FRAMEWORK_LAYERCUDA_H
