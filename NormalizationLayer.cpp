//
// Created by Sheen156 on 6/12/2018.
//

#include "NormalizationLayer.h"
#include "statisTool.h"

NormalizationLayer::NormalizationLayer(const int id, const string name,Layer* preLayer):Layer(id,name, preLayer->m_tensorSize){
    m_type = "NormalizationLayer";
    setPreviousLayer(preLayer);
    m_epsilon = 1e-6;
}
NormalizationLayer::~NormalizationLayer(){

}

void NormalizationLayer::initialize(const string& initialMethod){
     //null
    return;
}

// Y = (X-mu)/sigma
// sigma = sigma + m_epsilon,   avoid sigma == 0
// dL/dX = dL/dY * dY/dX = dL/dY * (1/sigma)
void NormalizationLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayerPointer->m_pYTensor;
    float mean = average(&X);
    float sigma = sqrt(variance(&X));
    for (long i=0; i< m_tensorSize; ++i){
        Y[i] = (X[i]- mean)/(sigma+m_epsilon);
    }
}
void NormalizationLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayerPointer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayerPointer->m_pYTensor;
    float sigma = sqrt(variance(&X));
    for(long i=0; i< m_tensorSize; ++i){
        dX[i] = dY[i]/(sigma+m_epsilon);
    }
}
void NormalizationLayer::updateParameters(const float lr, const string& method){
    return;
}