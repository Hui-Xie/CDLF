//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <LinearLayer.h>
#include "statisTool.h"


LinearLayer::LinearLayer(const int id, const string& name, Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
   m_type = "LinearLayer";
   addPreviousLayer(prevLayer);
   m_pKTensor =   new Tensor<float>(prevLayer->m_tensorSize);
   m_pdKTensor =  new Tensor<float>(prevLayer->m_tensorSize);
   m_pBTensor  =   new Tensor<float>(prevLayer->m_tensorSize);
   m_pdBTensor =  new Tensor<float>(prevLayer->m_tensorSize);
}

LinearLayer::~LinearLayer() {
    if (nullptr != m_pKTensor) {
        delete m_pKTensor;
        m_pKTensor = nullptr;
    }
    if (nullptr != m_pdKTensor) {
        delete m_pdKTensor;
        m_pdKTensor = nullptr;
    }

    if (nullptr != m_pBTensor) {
        delete m_pBTensor;
        m_pBTensor = nullptr;
    }
    if (nullptr != m_pdBTensor) {
        delete m_pdBTensor;
        m_pdBTensor = nullptr;
    }
}


void LinearLayer::initialize(const string& initialMethod){
  int N = m_pBTensor->getLength();
  generateGaussian(m_pKTensor, 0, sqrt(1.0/N));
  generateGaussian(m_pBTensor, 0, sqrt(1.0/N));
}

void LinearLayer::zeroParaGradient(){
    if (nullptr != m_pdKTensor) m_pdKTensor->zeroInitialize();
    if (nullptr != m_pdBTensor) m_pdBTensor->zeroInitialize();
}

//Y_i = K_i*X_i + B_i    for each element
void LinearLayer::forward(){
    const int N = m_pKTensor->getLength();
    for (int i=0; i<N; ++i){
        m_pYTensor->e(i) = m_pKTensor->e(i) * m_prevLayer->m_pYTensor->e(i) + m_pBTensor->e(i);
    }
}

/*
 *  Y_i = K_i*X_i + B_i    for each element
 *  dL/dX = dL/dY * K_i    Where L is Loss
 *  dL/dk = dL/dY * X_i
 *  dL/db = dL/dY
 */
void LinearLayer::backward(bool computeW, bool computeX){
    const int N = m_pKTensor->getLength();
    if (computeW) {
        for (int i=0; i<N; ++i){
            m_pdKTensor->e(i) += m_pdYTensor->e(i) * m_prevLayer->m_pYTensor->e(i);
        }
        *m_pdBTensor += *m_pdYTensor;
    }
    if (computeX){
        for (int i=0; i<N; ++i){
            m_prevLayer->m_pdYTensor->e(i) += m_pdYTensor->e(i) * m_pKTensor->e(i);
        }
    }
}

void LinearLayer::updateParameters(const float lr, const string& method, const int batchSize){
    if ("sgd" == method){
        *m_pBTensor -=  (*m_pdBTensor)*(lr/batchSize);
        *m_pKTensor -=  (*m_pdKTensor)*(lr/batchSize);
    }
}

int LinearLayer::getNumParameters(){
     return 2*m_pBTensor->getLength();
}

void LinearLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/B.csv";
    m_pBTensor->save(filename);

    filename= layerDir + "/K.csv";
    m_pKTensor->save(filename);
}

void LinearLayer::load(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)){
        initialize("Xavier");
        return;
    }
    else{
        filename= layerDir + "/B.csv";
        m_pBTensor->load(filename);

        filename= layerDir + "/K.csv";
        m_pKTensor->load(filename);
    }
}

void LinearLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void LinearLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
