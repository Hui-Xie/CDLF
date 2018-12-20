//
// Created by Hui Xie on 6/7/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <ReLU.h>

#include "ReLU.h"

//ReLU has just one previous layer.

ReLU::ReLU(const int id, const string& name,Layer* prevLayer, const float k): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "ReLU";
    m_k = k;
    addPreviousLayer(prevLayer);
}

ReLU::~ReLU(){

}

// Y = X if X >= m_k;
// Y = 0 if x >  m_k;
// dL/dx = dL/dy * dy/dx = dL/dy if X>=m_k;
// dL/dx = 0 if X < m_k
void ReLU::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int N = Y.getLength();
    for (int i=0; i< N; ++i){
       if (X.e(i) >= m_k ) Y.e(i) = X.e(i);
       else Y.e(i) = 0;
    }
}
void ReLU::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        Tensor<float>& X = *m_prevLayer->m_pYTensor;
        const int N = dY.getLength();
        for(int i=0; i< N; ++i){
            if (X.e(i) >= m_k) dX.e(i) += dY.e(i);
            // all dX.e(i) = 0 in zeroDYTensor() method in each iteration.
        }
    }
}
void ReLU::initialize(const string& initialMethod){
    //null
}

void ReLU::zeroParaGradient(){
    //null
}

void ReLU::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int  ReLU::getNumParameters(){
    return 0;
}

void ReLU::save(const string &netDir) {
   //null
}

void ReLU::load(const string &netDir) {
  //null
}

void ReLU::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, previousLayerIDs, outputTensorSize, filterSize, numFilter, FilterStride, startPosition, \r\n";
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, m_k, "{}");
}

void ReLU::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, k=%f, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), m_k, vector2Str(m_tensorSize).c_str());
}
