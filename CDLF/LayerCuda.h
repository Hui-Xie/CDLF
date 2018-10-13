//
// Created by Hui Xie on 10/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_LAYERCUDA_H
#define CDLF_FRAMEWORK_LAYERCUDA_H

void cudaSigmoidDerivative(float* pX, float* pdY, const int k, float* pdX, const long N);

void cudaSigmoid(float* pX, float* pY, const int k, const long N);


void cudaCrossEntropyGradient(float* pX, float* pGTX, float* pdX, const float epsilon, const long N);

#endif //CDLF_FRAMEWORK_LAYERCUDA_H
