//
// Created by Hui Xie on 9/10/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <ExponentialLayer.h>

#include "ExponentialLayer.h"

/* y_i = exp(x_i)
 *
 * */


ExponentialLayer::ExponentialLayer(const int id, const string& name,Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "ExponentialLayer";
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
    dX += dY.hadamard(X.expon());

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

void ExponentialLayer::save(const string &netDir) {
   //null
}

void ExponentialLayer::load(const string &netDir) {
  //null
}

void ExponentialLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void ExponentialLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s: (%s, id=%d): PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
