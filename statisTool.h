//
// Created by Sheen156 on 6/7/2018.
//

#ifndef RL_NONCONVEX_STATISTOOL_H
#define RL_NONCONVEX_STATISTOOL_H

#include  <blaze/Math.h>
using namespace blaze;

void generateGaussian(DynamicVector<float>* yVector,const float mu, const float sigma );
void xavierInitialize(DynamicMatrix<float>* pW);




#endif //RL_NONCONVEX_STATISTOOL_H
