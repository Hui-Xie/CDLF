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


void MatrixLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        //*m_pW -= (*m_pdW) * (lr / batchSize);
        matAdd(1.0, m_pW, -(lr / batchSize), m_pdW, m_pW);
        //*m_pB -= (*m_pdB) * (lr / batchSize);
        matAdd(1.0, m_pB, -(lr / batchSize), m_pdB, m_pB);
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

void MatrixLayer::initializeLRs(const float lr) {

}

void MatrixLayer::updateLRs(const float deltaLoss, const int batchSize) {

}

void MatrixLayer::updateParameters(const string &method, const int batchSize) {

}

