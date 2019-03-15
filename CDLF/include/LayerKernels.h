//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERKERNELS_H
#define CDLF_FRAMEWORK_LAYERKERNELS_H

__global__ void deviceSigmoidDerivative(const float* __restrict__  pX, const float* __restrict__  pdY, float* pdX, const int k, const int N);

__global__ void deviceSigmoid(const float* __restrict__  pX, float* pY, const int k, const int N);

__global__ void deviceCrossEntropyGradient(const float* __restrict__  pX, const float* __restrict__  pGTX, float* pdX, const float epsilon, const int N);


//C = A where A and C has different value type
__global__ void deviceElementCopy(const unsigned char* __restrict__  pA,float* pC, const int N);

//C = A if A>=0; C =0 else
__global__ void deviceRelu(const float* __restrict__  pA,float* pC, const int N);

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
__global__ void deviceReluDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int N);

__global__ void deviceSoftmax(const float* __restrict__  pX, float* pY, const int nSoftmax, const int N);

__global__ void deviceSoftmaxDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int nSoftmax, const int N);

//C = A*F in convolution
__global__ void deviceConvLayerForward(const float* pA, const int* pADimsSpan, const float* pF, const int* pFDimsSpan, const int filterSize, const int NFilter,
                                      const int stride, float* pC, const int* pCDimsSpan, const int* pNonZeroIndex, const int CDimsSize, const int N);

#endif //CDLF_FRAMEWORK_LAYERKERNELS_H
