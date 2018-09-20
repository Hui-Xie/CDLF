//
// Created by Hui Xie on 9/13/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "SubTensorLayer.h"


SubTensorLayer::SubTensorLayer(const int id, const string &name, Layer *prevLayer, const vector<long>& start, const vector<long>& span)
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
    long N = Y.getLength();
    for (long i=0; i< N; ++i){
        Y.e(i) = X.e(m_start+ Y.offset2Index(i));
    }
}

void SubTensorLayer::backward(bool computeW){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
    long N = dY.getLength();
    for(long i=0; i< N; ++i){
        dX.e(m_start+ dY.offset2Index(i)) += dY.e(i);
    }
}

void SubTensorLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}