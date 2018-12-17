//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <SigmoidLayer.h>

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
 * dL/dx = dL/dY * dY/dx = dL/dy *Y*(1- Y/k);
 * */
void SigmoidLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    int N = Y.getLength();
    for (int i=0; i< N; ++i){
        float exp_x = exp(-X.e(i));
        Y.e(i) = m_k/(1+exp_x);
    }
}
void SigmoidLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& Y = *m_pYTensor;
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        int N = dY.getLength();

        // use method: dL/dx = dL/dY * dY/dx = dL/dY *k * exp(x)/(1 +exp(x))^2
        /*Tensor<float>& X = *m_prevLayer->m_pYTensor;
        for(int i=0; i< N; ++i){
            float  expx = exp(X.e(i));
            dX.e(i) += dY.e(i)*m_k*expx/pow(1+expx,2);
        }*/

        //use method:dL/dx = dL/dY * dY/dx = dL/dy *Y*(1- Y/k);
        for(int i=0; i< N; ++i){
            dX.e(i) = dY.e(i)*Y.e(i)*(1-Y.e(i)/m_k);
        }
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

int  SigmoidLayer::getNumParameters(){
    return 0;
}

void SigmoidLayer::save(const string &netDir) {
  //null
}

void SigmoidLayer::load(const string &netDir) {
  //null
}

void SigmoidLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, m_k, "{}");
}

void SigmoidLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, k=%d, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), m_k, vector2Str(m_tensorSize).c_str());
}
