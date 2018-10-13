//
// Created by Hui Xie on 9/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "ExponentialLayer.h"

/* y_i = exp(x_i)
 *
 * */


ExponentialLayer::ExponentialLayer(const int id, const string& name,Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "Exponential";
    addPreviousLayer(prevLayer);
}

ExponentialLayer::~ExponentialLayer(){

}

// Y_i = exp(X_i)
// dL/dx = dL/dy * dy/dx = dL/dy * exp(x_i)
void ExponentialLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = Y.getLength();
    Y = X.expon();
}
void ExponentialLayer::backward(bool computeW){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = dY.getLength();
    for(long i=0; i< N; ++i){
        dX.e(i) += dY.e(i)*exp(X.e(i));
    }
}
void ExponentialLayer::initialize(const string& initialMethod){
    //null
}

void ExponentialLayer::zeroParaGradient(){
    //null
}

void ExponentialLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

long ExponentialLayer::getNumParameters(){
    return 0;
}