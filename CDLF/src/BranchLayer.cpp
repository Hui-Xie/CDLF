//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <BranchLayer.h>

#include "BranchLayer.h"

BranchLayer::BranchLayer(const int id, const string& name, Layer *prevLayer): Layer(id, name, prevLayer->m_tensorSize)
{
    m_type = "BranchLayer";
    addPreviousLayer(prevLayer);
}

BranchLayer::~BranchLayer()
{
    //null
}

void BranchLayer::initialize(const string& initialMethod){
    //null
}

void BranchLayer::zeroParaGradient(){
    //null
}

void BranchLayer::forward(){
    *m_pYTensor = *(m_prevLayer->m_pYTensor);
}

void BranchLayer::backward(bool computeW){
    // m_pdYTensor has been initialize to zero
    *(m_prevLayer->m_pdYTensor) += *(m_pdYTensor);
}

void BranchLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}

void BranchLayer::addNextLayer(Layer* nextLayer){
    if (nullptr != nextLayer){
        m_nextLayers.push_back(nextLayer);
    }
}

long BranchLayer::getNumParameters(){
    return 0;

}

void BranchLayer::save(const string &netDir) {
   //null
}

void BranchLayer::load(const string &netDir) {
    //null
}

void BranchLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}
