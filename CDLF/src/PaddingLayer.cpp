//
// Created by Hui Xie on 11/28/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "PaddingLayer.h"


PaddingLayer::PaddingLayer(const int id, const string &name, Layer *prevLayer, const vector<int>& tensorSize)
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
    int N = X.getLength();
    for (int i=0; i< N; ++i){
        Y.e(m_start+ X.offset2Index(i)) = X.e(i);
    }
}

void PaddingLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
        int N = dX.getLength();
        for(int i=0; i< N; ++i){
            dX.e(i) += dY.e(m_start+ dX.offset2Index(i));
        }
    }
}

void PaddingLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
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
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, vector2Str(m_start).c_str());

}

void PaddingLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, StartPosition=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_start).c_str(), vector2Str(m_tensorSize).c_str());
}
