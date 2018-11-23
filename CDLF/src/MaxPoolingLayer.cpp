//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <MaxPoolingLayer.h>

#include "MaxPoolingLayer.h"


MaxPoolingLayer::MaxPoolingLayer(const int id, const string &name, Layer *prevLayer, const vector<long> &filterSize,
                                  const int stride)
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
        m_tensorSize[i] = (m_tensorSize[i] - m_filterSize[i]) / m_stride + 1;
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
    const long N = length(m_tensorSize);
    Tensor<float>* pSubX = new Tensor<float>(m_filterSize);
    for (long i=0; i< N; ++i){
        vector<long> index = m_pYTensor->offset2Index(i);
        m_prevLayer->m_pYTensor->subTensorFromTopLeft(index*m_stride, pSubX);
        m_pYTensor->e(i) = pSubX->max();
    }
    if (nullptr != pSubX){
        delete pSubX;
        pSubX = nullptr;
    }
}

// Y_i = max(X_i) in filterSize range
// dL/dX_i = dL/dY * 1 when Xi = max; 0 otherwise;
void MaxPoolingLayer::backward(bool computeW) {
    const long N = length(m_tensorSize);
    Tensor<float>* pSubX = new Tensor<float>(m_filterSize);
    for (long i=0; i< N; ++i){
        vector<long> index = m_pdYTensor->offset2Index(i);
        vector<long> indexX = index*m_stride;
        m_prevLayer->m_pYTensor->subTensorFromTopLeft(indexX, pSubX);
        const long maxPos = pSubX->maxPosition();
        vector<long> maxIndex = pSubX->offset2Index(maxPos);
        m_prevLayer->m_pdYTensor->e(indexX + maxIndex) += m_pdYTensor->e(i);
    }
    if (nullptr != pSubX){
        delete pSubX;
        pSubX = nullptr;
    }
}

void MaxPoolingLayer::updateParameters(const float lr, const string &method, const int batchSize) {
    //null
}

long  MaxPoolingLayer::getNumParameters(){
    return 0;
}

void MaxPoolingLayer::save(const string &netDir) {
//null
}

void MaxPoolingLayer::load(const string &netDir) {
//null
}

void MaxPoolingLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), vector2Str(m_filterSize).c_str(), 0, m_stride, "{}");
}

void MaxPoolingLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s: (%s, id=%d): PrevLayer=%s, FilterSize=%s, NumOfFilter=1, Stide=%d, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_filterSize).c_str(), m_stride, vector2Str(m_tensorSize).c_str());
}


