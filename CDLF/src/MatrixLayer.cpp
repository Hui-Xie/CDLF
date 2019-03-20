//
// Created by Hui Xie on 1/19/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include "statisTool.h"
#include <iostream>
#include <MatrixLayer.h>
#include "TensorBlas.h"



using namespace std;

// Y = X*W +B or Y = W*X + B

MatrixLayer::MatrixLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize, const vector<int>& tensorSize)
        : Layer(id, name,tensorSize)
{
    if (2 != filterSize.size() || 2 != prevLayer->m_tensorSize.size()){
        cout<<"Error: MatrixLayer input parameter error. The filterSize and preLayer must be 2D matrix."<<endl;
        return;
    }

    m_type = "MatrixLayer";
    m_pW = new Tensor<float>(filterSize);
    m_pdW = new Tensor<float>(filterSize);
    m_pB = new Tensor<float>(m_pYTensor->getDims());
    m_pdB = new Tensor<float>(m_pYTensor->getDims());

    m_pWM = nullptr;
    m_pBM = nullptr;
    m_pWR = nullptr;
    m_pBR = nullptr;

    addPreviousLayer(prevLayer);
}

MatrixLayer::~MatrixLayer() {
    if (nullptr != m_pW) {
        delete m_pW;
        m_pW = nullptr;
    }
    if (nullptr != m_pB) {
        delete m_pB;
        m_pB = nullptr;
    }
    if (nullptr != m_pdW) {
        delete m_pdW;
        m_pdW = nullptr;
    }
    if (nullptr != m_pdB) {
        delete m_pdB;
        m_pdB = nullptr;
    }
}

void MatrixLayer::initialize(const string &initialMethod) {
    if ("Xavier" == initialMethod) {
        xavierInitialize(m_pW);
        xavierInitialize(m_pB);
    } else {
        cout << "Error: Initialize Error in MatrixLayer." << endl;
    }
}

void MatrixLayer::zeroParaGradient() {
    if (nullptr != m_pdW) {
        m_pdW->zeroInitialize();
    }
    if (nullptr != m_pdB) {
        m_pdB->zeroInitialize();
    }
}


void MatrixLayer::updateParameters(const string& method, Optimizer* pOptimizer) {
    if ("SGD" == method) {
        //*m_pW -= (*m_pdW) * (lr / batchSize);
        //matAdd(1.0, m_pW, -lr, m_pdW, m_pW);
        //*m_pB -= (*m_pdB) * (lr / batchSize);
        //matAdd(1.0, m_pB, -lr, m_pdB, m_pB);

        SGDOptimizer* sgdOptimizer = (SGDOptimizer*) pOptimizer;
        sgdOptimizer->sgd(m_pdW, m_pW);
        sgdOptimizer->sgd(m_pdB, m_pB);
    }
    else if ("Adam" == method){
        AdamOptimizer* adamOptimizer = (AdamOptimizer*) pOptimizer;
        adamOptimizer->adam(m_pWM, m_pWR, m_pdW, m_pW);
        adamOptimizer->adam(m_pBM, m_pBR, m_pdB, m_pB);
    }
    else{
        cout<<"Error: incorrect optimizer name."<<endl;
    }
}

int MatrixLayer::getNumParameters(){
    return m_pW->getLength() + m_pB->getLength();
}

void MatrixLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/W.csv";
    m_pW->save(filename);

    filename= layerDir + "/B.csv";
    m_pB->save(filename);
}

void MatrixLayer::load(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)){
        initialize("Xavier");
        return;
    }
    else{
        filename= layerDir + "/W.csv";
        if (!m_pW->load(filename)){
            xavierInitialize(m_pW);
        }

        filename= layerDir + "/B.csv";
        if (!m_pB->load(filename)){
            xavierInitialize(m_pB);
        }
    }
}

void MatrixLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), vector2Str(m_pW->getDims()).c_str(), "{}", 0, 0, "{}");
}

void MatrixLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, FilterSize=%s, Stride=%s, NumOfFilter=%d, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_pW->getDims()).c_str(), "{}", 0, vector2Str(m_tensorSize).c_str());
}

/*
void MatrixLayer::initializeLRs(const float lr) {

}

void MatrixLayer::updateLRs(const float deltaLoss) {

}

*/

void MatrixLayer::averageParaGradient(const int batchSize) {
    int N = m_pdW->getLength();
    cblas_saxpby(N, 1.0/batchSize, m_pdW->getData(), 1, 0, m_pdW->getData(), 1);
    N = m_pdB->getLength();
    cblas_saxpby(N, 1.0/batchSize, m_pdB->getData(), 1, 0, m_pdB->getData(), 1);
}

void MatrixLayer::allocateOptimizerMem(const string method) {
    if ("Adam" == method){
        m_pWM = new Tensor<float> (m_pW->getDims());  //1st moment
        m_pBM = new Tensor<float> (m_pB->getDims());
        m_pWR = new Tensor<float> (m_pW->getDims());  //2nd moment
        m_pBR = new Tensor<float> (m_pB->getDims());

        m_pWM->zeroInitialize();
        m_pBM->zeroInitialize();
        m_pWR->zeroInitialize();
        m_pBR->zeroInitialize();
    }
}

void MatrixLayer::freeOptimizerMem() {
    if (nullptr != m_pWM) {
        delete m_pWM;
        m_pWM = nullptr;
    }
    if (nullptr != m_pBM) {
        delete m_pBM;
        m_pBM = nullptr;
    }
    if (nullptr != m_pWR) {
        delete m_pWR;
        m_pWR = nullptr;
    }
    if (nullptr != m_pBR) {
        delete m_pBR;
        m_pBR = nullptr;
    }
}

