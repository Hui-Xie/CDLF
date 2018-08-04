//
// Created by Hui Xie on 6/7/2018.
//

#ifndef RL_NONCONVEX_STATISTOOL_H
#define RL_NONCONVEX_STATISTOOL_H

//#include  <blaze/Math.h>
#include "Tensor.h"
//using namespace blaze;

void generateGaussian(Tensor<float>* yTensor,const float mu, const float sigma  );
void xavierInitialize(Tensor<float>* pW);




#endif //RL_NONCONVEX_STATISTOOL_H
