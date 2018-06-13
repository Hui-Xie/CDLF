//
// Created by Sheen156 on 6/12/2018.
//

#include "NormalizationLayer.h"
#include "statisTool.h"

NormalizationLayer::NormalizationLayer(Layer* preLayer):Layer(preLayer->m_width){
    m_type = "NormalizationLayer";
    m_prevLayerPointer = preLayer;
    m_prevLayerPointer->m_nextLayerPointer = this;
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
    DynamicVector<float>& Y = *m_pYVector;
    DynamicVector<float>& X = *m_prevLayerPointer->m_pYVector;
    float mean = average(&X);
    float sigma = sqrt(variance(&X));
    for (long i=0; i< m_width; ++i){
        Y[i] = (X[i]- mean)/(sigma+m_epsilon);
    }
}
void NormalizationLayer::backward(){
    DynamicVector<float>& dY = *m_pdYVector;
    DynamicVector<float>& dX = *m_prevLayerPointer->m_pdYVector;
    DynamicVector<float>& X = *m_prevLayerPointer->m_pYVector;
    float sigma = sqrt(variance(&X));
    for(long i=0; i< m_width; ++i){
        dX[i] = dY[i]/(sigma+m_epsilon);
    }
}
void NormalizationLayer::updateParameters(const float lr, const string& method){
    return;
}