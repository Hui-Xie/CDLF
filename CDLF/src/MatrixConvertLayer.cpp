//
// Created by Hui Xie on 1/16/2019.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "MatrixConvertLayer.h"
#include "statisTool.h"
#include <iostream>
#include <MatrixConvertLayer.h>
#include <TensorBlas.h>


using namespace std;

// y = W*x
// where y is m*n output matrix;
//       x is k*n input matrix;
//       W is m*k dimensional matrix
//       B is same size with y: m*n
MatrixConvertLayer::MatrixConvertLayer(const int id, const string& name, Layer* prevLayer, const vector<int>& filterSize)
        : Layer(id, name,{filterSize[0], prevLayer->m_tensorSize[1]})
{
    if (2 != filterSize.size() || 2 != prevLayer->m_tensorSize.size()){
        cout<<"Error: MatrixCovnertLayer input parameter error. The filterSize and preLayer must be 2D matrix."<<endl;
        return;
    }

    m_type = "MatrixConvertLayer";
    m_pW = new Tensor<float>(filterSize);
    m_pdW = new Tensor<float>(filterSize);
    m_pB = new Tensor<float>(m_pYTensor->getDims());
    m_pdB = new Tensor<float>(m_pYTensor->getDims());
    addPreviousLayer(prevLayer);
}

MatrixConvertLayer::~MatrixConvertLayer() {
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

void MatrixConvertLayer::initialize(const string &initialMethod) {
    if ("Xavier" == initialMethod) {
        xavierInitialize(m_pW);
        xavierInitialize(m_pB);
    } else {
        cout << "Error: Initialize Error in MatrixConvertLayer." << endl;
    }
}

void MatrixConvertLayer::zeroParaGradient() {
    if (nullptr != m_pdW) {
        m_pdW->zeroInitialize();
    }
    if (nullptr != m_pdB) {
        m_pdB->zeroInitialize();
    }
}

void MatrixConvertLayer::forward() {
    *m_pYTensor = *m_pB;
    gemm(1.0, false, m_pW, false, m_prevLayer->m_pYTensor, 1, m_pYTensor);
}

//   y = W*x +B
//  dL/dW = dL/dy * dy/dW = dL/dy * x'
//  dL/dB = dL/dy * dy/dB = dL/dy
//  dL/dx = dL/dy * dy/dx = W' * dL/dy
void MatrixConvertLayer::backward(bool computeW, bool computeX) {
    Tensor<float> &dLdy = *m_pdYTensor;
    if (computeW){
        gemm(1.0, false, &dLdy, true, m_prevLayer->m_pYTensor, 1, m_pdW);
        matAdd(1, m_pdB, 1, &dLdy, m_pdB);
    }
    if (computeX){
        gemm(1.0, true, m_pW, false, &dLdy, 1, m_prevLayer->m_pdYTensor);
    }
}

void MatrixConvertLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    if ("sgd" == method) {
        //*m_pW -= (*m_pdW) * (lr / batchSize);
        matAdd(1.0, m_pW, -(lr / batchSize), m_pdW, m_pW);
        //*m_pB -= (*m_pdB) * (lr / batchSize);
        matAdd(1.0, m_pB, -(lr / batchSize), m_pdB, m_pB);
    }
}

int MatrixConvertLayer::getNumParameters(){
    return m_pW->getLength() + m_pB->getLength();
}

void MatrixConvertLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/W.csv";
    m_pW->save(filename);

    filename= layerDir + "/B.csv";
    m_pB->save(filename);
}

void MatrixConvertLayer::load(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)){
        initialize("Xavier");
        return;
    }
    else{
        filename= layerDir + "/W.csv";
        m_pW->load(filename);

        filename= layerDir + "/B.csv";
        m_pB->load(filename);
    }
}

void MatrixConvertLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), vector2Str(m_pW->getDims()).c_str(), 0, 0, "{}");
}

void MatrixConvertLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, FilterSize=%s, NumOfFilter=%d, Stide=%d, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_pW->getDims()).c_str(), 0, 0, vector2Str(m_tensorSize).c_str());
}

