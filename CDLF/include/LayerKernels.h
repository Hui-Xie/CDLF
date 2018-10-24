//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERKERNELS_H
#define CDLF_FRAMEWORK_LAYERKERNELS_H

__global__ void deviceSigmoidDerivative(const float* __restrict__  pX, const float* __restrict__  pdY, float* pdX, const int k, const long N);

__global__ void deviceSigmoid(const float* __restrict__  pX, float* pY, const int k, const long N);

__global__ void deviceCrossEntropyGradient(const float* __restrict__  pX, const float* __restrict__  pGTX, float* pdX, const float epsilon, const long N);


//C = A where A and C has different value type
__global__ void deviceElementCopy(const unsigned char* __restrict__  pA,float* pC, const long N);

//C = A if A>=0; C =0 else
__global__ void deviceRelu(const float* __restrict__  pA,float* pC, const long N);

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
__global__ void deviceReluDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const long N);

__global__ void deviceSoftmax(const float* __restrict__  pX, float* pY, const int nSoftmax, const long N);

__global__ void deviceSoftmaxDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int nSoftmax, const long N);

//C = A*F in convolution
__global__ void deviceConvLayerForward(const float* pA, const long* pADimsSpan, const float* pF, const long* pFDimsSpan, const int spanSize, const long NFilter,
                                      const int stride, float* pC, const long* pCDimsSpan, const long N);

#endif //CDLF_FRAMEWORK_LAYERKERNELS_H
