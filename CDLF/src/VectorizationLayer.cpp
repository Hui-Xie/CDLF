//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <VectorizationLayer.h>

#include "VectorizationLayer.h"


VectorizationLayer::VectorizationLayer(const int id, const string& name,Layer* prevLayer)
   : Layer(id,name, {prevLayer->m_pYTensor->getLength(),1}){
    m_type = "VectorizationLayer";
    addPreviousLayer(prevLayer);
}

VectorizationLayer::~VectorizationLayer(){

}

void VectorizationLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    Y = X.vectorize();
}
void VectorizationLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        dX += dY.reshape(dX.getDims());
    }
}
void VectorizationLayer::initialize(const string& initialMethod){
    //null
}

void VectorizationLayer::zeroParaGradient(){
    //null
}

void VectorizationLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

long  VectorizationLayer::getNumParameters(){
    return 0;
}

void VectorizationLayer::save(const string &netDir) {
//null
}

void VectorizationLayer::load(const string &netDir) {
//null
}

void VectorizationLayer::saveStructLine(FILE *pFile) {
//const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void VectorizationLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s: (%s, id=%d): PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
