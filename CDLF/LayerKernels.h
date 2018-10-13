//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERKERNELS_H
#define CDLF_FRAMEWORK_LAYERKERNELS_H

__global__ void deviceSigmoidDerivative(float* pX, float* pdY, const int k, float* pdX, const long N);

__global__ void deviceSigmoid(float* pX, float* pY, const int k, const long N);

__global__ void deviceCrossEntropyGradient(float* pX, float* pGTX, float* pdX, const float epsilon, const long N);


//C = A where A and C has different value type
__global__ void deviceElementCopy(unsigned char* pA,float* pC, const long N);

//C = A if A>=0; C =0 else
__global__ void deviceRelu(float* pA,float* pC, const long N);

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
__global__ void deviceReluDerivative(float* pX,float* pdY, float* pdX, const long N);

__global__ void deviceSoftmax(float* pX, float* pY, const int nSoftmax, const long N);

__global__ void deviceSoftmaxDerivative(float* pX,float* pdY, float* pdX, const int nSoftmax, const long N);

#endif //CDLF_FRAMEWORK_LAYERKERNELS_H
