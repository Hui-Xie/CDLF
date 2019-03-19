//
// Created by Hui Xie on 11/28/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include <PaddingLayer.h>

#include "PaddingLayer.h"


PaddingLayer::PaddingLayer(const int id, const string &name, Layer *prevLayer, const vector<int>& tensorSize, const float initialValue)
        :Layer(id,name, tensorSize)
{
    m_type = "PaddingLayer";
    addPreviousLayer(prevLayer);
    const int dim = m_tensorSize.size();
    for (int i=0; i< dim; ++i){
        if (tensorSize[i] < m_prevLayer->m_tensorSize[i]){
            cout<<"Error: previous Layer's tensorsize should be less than current tensorSize."<<endl;
        }
    }
    m_start = (m_tensorSize - m_prevLayer->m_tensorSize)/2;
    m_initialValue = initialValue;
}

PaddingLayer::~PaddingLayer(){

}

void PaddingLayer::initialize(const string& initialMethod){
    //null
}

void PaddingLayer::zeroParaGradient(){
    //null
}

void PaddingLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int N = X.getLength();
    for (int i=0; i< N; ++i){
        Y.e(m_start+ X.offset2Index(i)) = X.e(i);
    }
}

void PaddingLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
        const int N = dX.getLength();
        for(int i=0; i< N; ++i){
            dX.e(i) += dY.e(m_start+ dX.offset2Index(i));
        }
    }
}



int  PaddingLayer::getNumParameters(){
    return 0;
}

void PaddingLayer::save(const string &netDir) {
//null
}

void PaddingLayer::load(const string &netDir) {
    //null
}

void PaddingLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, vector2Str(m_start).c_str());

}

void PaddingLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, StartPosition=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_start).c_str(), vector2Str(m_tensorSize).c_str());
}
/*
void PaddingLayer::initializeLRs(const float lr) {

}

void PaddingLayer::updateLRs(const float deltaLoss) {

}
*/
void PaddingLayer::updateParameters(const string& method, Optimizer* pOptimizer) {

}

void PaddingLayer::averageParaGradient(const int batchSize) {

}
