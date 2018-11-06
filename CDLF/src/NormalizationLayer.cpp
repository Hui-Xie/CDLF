//
// Created by Hui Xie on 6/12/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "NormalizationLayer.h"
#include "statisTool.h"
#include <math.h>       /* sqrt */
#include <NormalizationLayer.h>


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
    Y = (X-mean)/(sigma+m_epsilon);

}
void NormalizationLayer::backward(bool computeW){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
    Tensor<float>& X =  *(m_prevLayer->m_pYTensor);
    float sigma = sqrt(X.variance());
    dX += dY/(sigma+m_epsilon);
}
void NormalizationLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}

long  NormalizationLayer::getNumParameters(){
    return 0;
}

void NormalizationLayer::save(const string &netDir) {

}

void NormalizationLayer::load(const string &netDir) {

}

void NormalizationLayer::saveArchitectLine(FILE *pFile) {

}
