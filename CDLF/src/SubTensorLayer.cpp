//
// Created by Hui Xie on 9/13/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <SubTensorLayer.h>

#include "SubTensorLayer.h"


SubTensorLayer::SubTensorLayer(const int id, const string &name, Layer *prevLayer, const vector<int>& start, const vector<int>& span)
               :Layer(id,name, span)
{
    m_type = "SubTensorLayer";
    addPreviousLayer(prevLayer);
    if (start.size() == span.size() && prevLayer->m_tensorSize.size() == start.size() && span <= prevLayer->m_tensorSize){
        m_start = start;
        m_span  = span;
    }
    else{
        cout<<"Error: previous Layer's tensorsize, start and span should have same size, and span <= preLayer's tensorSize."<<endl;
    }

}

SubTensorLayer::~SubTensorLayer(){

}

void SubTensorLayer::initialize(const string& initialMethod){
    //null
}

void SubTensorLayer::zeroParaGradient(){
    //null
}

void SubTensorLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    int N = Y.getLength();
    for (int i=0; i< N; ++i){
        Y.e(i) = X.e(m_start+ Y.offset2Index(i));
    }
}

void SubTensorLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
        int N = dY.getLength();
        for(int i=0; i< N; ++i){
            dX.e(m_start+ dY.offset2Index(i)) += dY.e(i);
        }
    }
}

void SubTensorLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}


int  SubTensorLayer::getNumParameters(){
    return 0;
}

void SubTensorLayer::save(const string &netDir) {
//null
}

void SubTensorLayer::load(const string &netDir) {
  //null
}

void SubTensorLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}",0, 0, vector2Str(m_start).c_str());

}

void SubTensorLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s,  PrevLayer=%s, StartPosition=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),   m_prevLayer->m_name.c_str(), vector2Str(m_start).c_str(), vector2Str(m_tensorSize).c_str());
}
