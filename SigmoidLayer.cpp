//
// Created by Hui Xie on 8/4/2018.
//

#include "SigmoidLayer.h"

SigmoidLayer::SigmoidLayer(const int id, const string& name,Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "SigmoidLayer";
    addPreviousLayer(prevLayer);
}

SigmoidLayer::~SigmoidLayer(){
  //null
}

/* Y = exp(x)/( 1+ exp(x)) in element-wise
 * dL/dx = dL/dY * dY/dx = dL/dY * exp(x)/(1 +exp(x))^2
 * */
void SigmoidLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = Y.getLength();
    for (long i=0; i< N; ++i){
        float expx = exp(X.e(i));
        Y.e(i) = expx/(1+expx);
    }
}
void SigmoidLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    long N = dY.getLength();
    for(long i=0; i< N; ++i){
        float  expx = exp(X.e(i));
        dX.e(i) += dY.e(i)*expx/pow(1+expx,2);
     }
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