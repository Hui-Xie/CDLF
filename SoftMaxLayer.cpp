//
// Created by Sheen156 on 7/28/2018.
//

#include "SoftMaxLayer.h"
#include <math.h>       /* exp */

SoftMaxLayer::SoftMaxLayer(const int id, const string& name,Layer* prevLayer):Layer(id,name, prevLayer->m_tensorSize) {
    m_type = "SoftMaxLayer";
    addPreviousLayer(prevLayer);
    m_sumExpX = 0.0;
}
SoftMaxLayer::~SoftMaxLayer(){

}

void SoftMaxLayer::initialize(const string& initialMethod){
    //null
}

void SoftMaxLayer::zeroParaGradient(){
    //null
}

// Y_i = exp(X_i)/ (\sum exp(x_i))
// dL/dX = dL/dY * dY/dX = dL/dY * exp(x_i)*(\sum exp(x_i)-exp(x_i))/(\sum exp(x_i))^2
void SoftMaxLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayers.front()->m_pYTensor;
    long N = Y.getLength();
    m_sumExpX = 0;
    for (long i=0; i< N; ++i){
        m_sumExpX += exp(X(i));
    }
    if (0 == m_sumExpX){
        cout<<"Error: SoftMax Layer m_sumExpX ==0 "<<endl;
        m_sumExpX = 1e-8;
    }
    for (long i=0; i< N; ++i){
        Y(i) = exp(X(i))/m_sumExpX;
    }
}
void SoftMaxLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayers.front()->m_pdYTensor;
    Tensor<float>& X = *m_prevLayers.front()->m_pYTensor;
    long N = dY.getLength();
    float m_sumExpX2 = m_sumExpX*m_sumExpX;
    for(long i=0; i< N; ++i){
        dX(i) = dY(i)*exp(X(i))*(m_sumExpX-exp(X(i)))/m_sumExpX2;
    }
}
void SoftMaxLayer::updateParameters(const float lr, const string& method){
    //Null
}