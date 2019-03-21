//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <LinearLayer.h>
#include "statisTool.h"


LinearLayer::LinearLayer(const int id, const string& name, Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
   m_type = "LinearLayer";
   addPreviousLayer(prevLayer);
   m_pK =   new Tensor<float>(prevLayer->m_tensorSize);
   m_pdK =  new Tensor<float>(prevLayer->m_tensorSize);
   m_pB  =   new Tensor<float>(prevLayer->m_tensorSize);
   m_pdB =  new Tensor<float>(prevLayer->m_tensorSize);

    m_pKM = nullptr;
    m_pBM = nullptr;
    m_pKR = nullptr;
    m_pBR = nullptr;
}

LinearLayer::~LinearLayer() {
    if (nullptr != m_pK) {
        delete m_pK;
        m_pK = nullptr;
    }
    if (nullptr != m_pdK) {
        delete m_pdK;
        m_pdK = nullptr;
    }

    if (nullptr != m_pB) {
        delete m_pB;
        m_pB = nullptr;
    }
    if (nullptr != m_pdB) {
        delete m_pdB;
        m_pdB = nullptr;
    }
}


void LinearLayer::initialize(const string& initialMethod){
  int N = m_pB->getLength();
  generateGaussian(m_pK, 0, 0.001);
  generateGaussian(m_pB, 0, 0.00001);
}

void LinearLayer::zeroParaGradient(){
    if (nullptr != m_pdK) m_pdK->zeroInitialize();
    if (nullptr != m_pdB) m_pdB->zeroInitialize();
}

//Y_i = K_i*X_i + B_i    for each element
void LinearLayer::forward(){
    const int N = m_pK->getLength();
    for (int i=0; i<N; ++i){
        m_pYTensor->e(i) = m_pK->e(i) * m_prevLayer->m_pYTensor->e(i) + m_pB->e(i);
    }
}

/*
 *  Y_i = K_i*X_i + B_i    for each element
 *  dL/dX = dL/dY * K_i    Where L is Loss
 *  dL/dk = dL/dY * X_i
 *  dL/db = dL/dY
 */
void LinearLayer::backward(bool computeW, bool computeX){
    const int N = m_pK->getLength();
    if (computeW) {
        for (int i=0; i<N; ++i){
            m_pdK->e(i) += m_pdYTensor->e(i) * m_prevLayer->m_pYTensor->e(i);
        }
        *m_pdB += *m_pdYTensor;
    }
    if (computeX){
        for (int i=0; i<N; ++i){
            m_prevLayer->m_pdYTensor->e(i) += m_pdYTensor->e(i) * m_pK->e(i);
        }
    }
}

void LinearLayer::updateParameters(Optimizer* pOptimizer){
    if ("SGD" == pOptimizer->m_type){
        SGDOptimizer* optimizer = (SGDOptimizer*) pOptimizer;
        optimizer->sgd(m_pdB, m_pB);
        optimizer->sgd(m_pdK, m_pK);
    }
    else if ("Adam" == pOptimizer->m_type){
        AdamOptimizer* optimizer = (AdamOptimizer*) pOptimizer;
        optimizer->adam(m_pKM, m_pKR, m_pdK, m_pK);
        optimizer->adam(m_pBM, m_pBR, m_pdB, m_pB);
    }
    else{
        cout<<"Error: incorrect optimizer name."<<endl;
    }
}

int LinearLayer::getNumParameters(){
     return 2*m_pB->getLength();
}

void LinearLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/B.csv";
    m_pB->save(filename);

    filename= layerDir + "/K.csv";
    m_pK->save(filename);
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
        if  (!m_pB->load(filename)){
            generateGaussian(m_pB, 0, 0.00001);
        }

        filename= layerDir + "/K.csv";
        if (! m_pK->load(filename)){
            generateGaussian(m_pK, 0, 0.001);
        }
    }
}

void LinearLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void LinearLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}

/*
void LinearLayer::initializeLRs(const float lr) {

}

void LinearLayer::updateLRs(const float deltaLoss) {

}
*/

void LinearLayer::averageParaGradient(const int batchSize) {
    int N = m_pdK->getLength();
    cblas_saxpby(N, 1.0/batchSize, m_pdK->getData(), 1, 0, m_pdK->getData(), 1);
    N = m_pdB->getLength();
    cblas_saxpby(N, 1.0/batchSize, m_pdB->getData(), 1, 0, m_pdB->getData(), 1);
}

void LinearLayer::allocateOptimizerMem(const string method) {
    if ("Adam" == method){
        m_pKM = new Tensor<float> (m_pK->getDims());  //1st moment
        m_pBM = new Tensor<float> (m_pB->getDims());
        m_pKR = new Tensor<float> (m_pK->getDims());  //2nd moment
        m_pBR = new Tensor<float> (m_pB->getDims());

        m_pKM->zeroInitialize();
        m_pBM->zeroInitialize();
        m_pKR->zeroInitialize();
        m_pBR->zeroInitialize();
    }
}

void LinearLayer::freeOptimizerMem() {
    if (nullptr != m_pKM) {
        delete m_pKM;
        m_pKM = nullptr;
    }
    if (nullptr != m_pBM) {
        delete m_pBM;
        m_pBM = nullptr;
    }
    if (nullptr != m_pKR) {
        delete m_pKR;
        m_pKR = nullptr;
    }
    if (nullptr != m_pBR) {
        delete m_pBR;
        m_pBR = nullptr;
    }
}
