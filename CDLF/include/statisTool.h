//
// Created by Hui Xie on 6/7/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_STATISTOOL_H
#define CDLF_FRAME_STATISTOOL_H

//#include  <blaze/Math.h>
#include "Tensor.h"
//using namespace blaze;

void generateGaussian(Tensor<float>* yTensor,const float mu, const float sigma  );
void xavierInitialize(Tensor<float>* pW);




#endif //CDLF_FRAME_STATISTOOL_H
