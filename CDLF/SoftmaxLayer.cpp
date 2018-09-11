//
// Created by Hui Xie on 7/28/2018.
//

#include "SoftmaxLayer.h"
#include <math.h>       /* exp */

SoftmaxLayer::SoftmaxLayer(const int id, const string& name,Layer* prevLayer):Layer(id,name, prevLayer->m_tensorSize) {
    m_type = "SoftmaxLayer";
    addPreviousLayer(prevLayer);
    m_sumExpX = 0.0;
}
SoftmaxLayer::~SoftmaxLayer(){

}

void SoftmaxLayer::initialize(const string& initialMethod){
    //null
}

void SoftmaxLayer::zeroParaGradient(){
    //null
}

// Y_i = exp(X_i)/ (\sum exp(x_i))
// dL/dX = dL/dY * dY/dX = dL/dY * exp(x_i)*(\sum exp(x_i)-exp(x_i))/(\sum exp(x_i))^2
void SoftmaxLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = X.getLength();
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
void SoftmaxLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = dY.getLength();
    float m_sumExpX2 = m_sumExpX*m_sumExpX;
    for(long i=0; i< N; ++i){
        dX(i) += dY(i)*exp(X(i))*(m_sumExpX-exp(X(i)))/m_sumExpX2;
    }
}
void SoftmaxLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //Null
}