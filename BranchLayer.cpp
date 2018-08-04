//
// Created by Hui Xie on 8/4/2018.
//

#include "BranchLayer.h"

BranchLayer::BranchLayer(const int id, const string& name, const vector<int>& tensorSize, Layer *prevLayer): Layer(id, name, tensorSize)
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

void BranchLayer::backward(){
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