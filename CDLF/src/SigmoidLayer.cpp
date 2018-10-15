//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "SigmoidLayer.h"

SigmoidLayer::SigmoidLayer(const int id, const string& name,Layer* prevLayer, const int k): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "SigmoidLayer";
    m_k = k;
    addPreviousLayer(prevLayer);
}

SigmoidLayer::~SigmoidLayer(){
  //null
}

/* Y = k/( 1+ exp(-x)) in element-wise
 * dL/dx = dL/dY * dY/dx = dL/dY *k * exp(x)/(1 +exp(x))^2
 * */
void SigmoidLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = Y.getLength();
#ifdef Use_GPU
    cudaSigmoid(X.getData(), Y.getData(),m_k, N);
#else
    for (long i=0; i< N; ++i){
        float exp_x = exp(-X.e(i));
        Y.e(i) = m_k/(1+exp_x);
    }
#endif
}
void SigmoidLayer::backward(bool computeW){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = dY.getLength();
#ifdef Use_GPU
    cudaSigmoidDerivative(X.getData(), dY.getData(), dX.getData(), m_k, N);
#else
    for(long i=0; i< N; ++i){
        float  expx = exp(X.e(i));
        dX.e(i) += dY.e(i)*m_k*expx/pow(1+expx,2);
     }
#endif

}
void SigmoidLayer::initialize(const string& initialMethod){
    //null
}

void SigmoidLayer::zeroParaGradient(){
    //null
}

void SigmoidLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

long  SigmoidLayer::getNumParameters(){
    return 0;
}