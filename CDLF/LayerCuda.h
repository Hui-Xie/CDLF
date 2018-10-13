//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERCUDA_H
#define CDLF_FRAMEWORK_LAYERCUDA_H

void cudaSigmoidDerivative(float* pX, float* pdY, const int k, float* pdX, const long N);

void cudaSigmoid(float* pX, float* pY, const int k, const long N);


void cudaCrossEntropyGradient(float* pX, float* pGTX, float* pdX, const float epsilon, const long N);

//C = A where A and C has different value type
void cudaElementCopy(unsigned char* pA,float* pC, const long N);

//C = A if A>=0; C =0 else
void cudaRelu(float* pA,float* pC, const long N);

// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
void cudaReluDerivative(float* pX,float* pdY, float* pdX, const long N);

#endif //CDLF_FRAMEWORK_LAYERCUDA_H