//
// Created by Hui Xie on 6/12/2018.
//

#include "NormalizationLayer.h"
#include "statisTool.h"
#include <math.h>       /* sqrt */

NormalizationLayer::NormalizationLayer(const int id, const string& name,Layer* prevLayer):Layer(id,name, prevLayer->m_tensorSize){
    m_type = "NormalizationLayer";
    addPreviousLayer(prevLayer);
    m_epsilon = 1e-6;
}
NormalizationLayer::~NormalizationLayer(){

}

void NormalizationLayer::initialize(const string& initialMethod){
     //null

}

void NormalizationLayer::zeroParaGradient(){
    //null
}

// Y = (X-mu)/sigma
// sigma = sigma + m_epsilon,   avoid sigma == 0
// dL/dX = dL/dY * dY/dX = dL/dY * (1/sigma)
void NormalizationLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    float mean = X.average();
    float sigma = sqrt(X.variance());
    long N = Y.getLength();
    for (long i=0; i< N; ++i){
        Y.e(i) = (X.e(i)- mean)/(sigma+m_epsilon);
    }
}
void NormalizationLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float>& X =  *(m_prevLayer->m_pYTensor);
    float sigma = sqrt(X.variance());
    long N = dY.getLength();
    for(long i=0; i< N; ++i){
       dX.e(i) += dY.e(i)/(sigma+m_epsilon);
    }
}
void NormalizationLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}