//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <MaxPoolingLayer.h>

#include "MaxPoolingLayer.h"


MaxPoolingLayer::MaxPoolingLayer(const int id, const string &name, Layer *prevLayer, const vector<int> &filterSize,
                                 const vector<int>& stride)
        : Layer(id, name, {}) {

    m_type = "MaxPoolingLayer";
    m_stride = stride;
    m_filterSize = filterSize;
    m_tensorSize = prevLayer->m_tensorSize; // this is an initial, not final, size

    int N = filterSize.size();
    m_OneFilterN = 1;
    for (int i = 0; i < N; ++i) {
        m_OneFilterN *= filterSize[i];
    }
    addPreviousLayer(prevLayer);
    constructY();
}

MaxPoolingLayer::~MaxPoolingLayer() {
    //null
}

void MaxPoolingLayer::constructY() {
    //get refined pYTensor size
    const int dim = m_tensorSize.size();
    for (int i = 0; i < dim; ++i) {
        m_tensorSize[i] = (m_tensorSize[i] - m_filterSize[i]) / m_stride[i] + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }
    allocateYdYTensor();
}

void MaxPoolingLayer::initialize(const string &initialMethod) {
    //null
}

void MaxPoolingLayer::zeroParaGradient() {
    //null
}

// Y_i = max(X_i) in filterSize range
void MaxPoolingLayer::forward() {
    const int N = length(m_tensorSize);
    Tensor<float>* pSubX = new Tensor<float>(m_filterSize);
    for (int i=0; i< N; ++i){
        vector<int> index = m_pYTensor->offset2Index(i);
        const vector<int> stride1 = vector<int>(m_filterSize.size(),1);
        m_prevLayer->m_pYTensor->subTensorFromTopLeft(index*m_stride, pSubX, stride1);
        m_pYTensor->e(i) = pSubX->max();
    }
    if (nullptr != pSubX){
        delete pSubX;
        pSubX = nullptr;
    }
}

// Y_i = max(X_i) in filterSize range
// dL/dX_i = dL/dY * 1 when Xi = max; 0 otherwise;
void MaxPoolingLayer::backward(bool computeW, bool computeX) {
    if (computeX){
        const int N = length(m_tensorSize);
        Tensor<float>* pSubX = new Tensor<float>(m_filterSize);
        for (int i=0; i< N; ++i){
            vector<int> index = m_pdYTensor->offset2Index(i);
            vector<int> indexX = index*m_stride;
            const vector<int> stride1 = vector<int>(m_filterSize.size(),1);
            m_prevLayer->m_pYTensor->subTensorFromTopLeft(indexX, pSubX, stride1);
            const int maxPos = pSubX->maxPosition();
            vector<int> maxIndex = pSubX->offset2Index(maxPos);
            m_prevLayer->m_pdYTensor->e(indexX + maxIndex) += m_pdYTensor->e(i);
        }
        if (nullptr != pSubX){
            delete pSubX;
            pSubX = nullptr;
        }
    }
}

void MaxPoolingLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    //null
}

int  MaxPoolingLayer::getNumParameters(){
    return 0;
}

void MaxPoolingLayer::save(const string &netDir) {
//null
}

void MaxPoolingLayer::load(const string &netDir) {
//null
}

void MaxPoolingLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), vector2Str(m_filterSize).c_str(), vector2Str(m_stride).c_str(), 0, 0, "{}");
}

void MaxPoolingLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, FilterSize=%s, Stride=%s, NumOfFilter=1, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),   m_prevLayer->m_name.c_str(), vector2Str(m_filterSize).c_str(), vector2Str(m_stride).c_str(), vector2Str(m_tensorSize).c_str());
}


