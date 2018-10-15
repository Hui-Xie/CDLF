//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERCUDA_H
#define CDLF_FRAMEWORK_LAYERCUDA_H

void cudaSigmoidDerivative(const float* __restrict__  pX, const float* __restrict__  pdY, float* pdX, const int k, const long N);

void cudaSigmoid(const float* __restrict__  pX, float* pY, const int k, const long N);


void cudaCrossEntropyGradient(const float* __restrict__  pX, const float* __restrict__  pGTX, float* pdX, const float epsilon, const long N);

//C = A where A and C has different value type
void cudaElementCopy(const unsigned char* __restrict__ pA,float* pC, const long N);

//C = A if A>=0; C =0 else
void cudaRelu(const float* __restrict__  pA,float* pC, const long N);

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
void cudaReluDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const long N);

void cudaSoftmax(const float* __restrict__  pX, float* pY, const int nSoftmax, const long N);

void cudaSoftmaxDerivative(const float* __restrict__  pX,const float* __restrict__  pdY, float* pdX, const int nSoftmax, const long N);

#endif //CDLF_FRAMEWORK_LAYERCUDA_H
