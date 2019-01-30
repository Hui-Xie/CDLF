//
// Created by Hui Xie on 8/3/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <IdentityLayer.h>

#include "IdentityLayer.h"

IdentityLayer::IdentityLayer(const int id, const string& name,Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "IdentityLayer";
    addPreviousLayer(prevLayer);
}

IdentityLayer::~IdentityLayer(){

}

// Y = X
// dL/dx = dL/dy * dy/dx = dL/dy
void IdentityLayer::forward(){
    *m_pYTensor = *m_prevLayer->m_pYTensor;
}
void IdentityLayer::backward(bool computeW, bool computeX){
   if(computeX){
      *m_prevLayer->m_pdYTensor += *m_pdYTensor;
   }
}
void IdentityLayer::initialize(const string& initialMethod){
    //null
}

void IdentityLayer::zeroParaGradient(){
    //null
}

void IdentityLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int IdentityLayer::getNumParameters(){
    return 0;
}

void IdentityLayer::save(const string &netDir) {
//null
}

void IdentityLayer::load(const string &netDir) {
//null
}

void IdentityLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void IdentityLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
