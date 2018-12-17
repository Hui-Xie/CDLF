//
// Created by Hui Xie on 7/28/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "SoftmaxLayer.h"
#include <math.h>       /* exp */
#include <SoftmaxLayer.h>


SoftmaxLayer::SoftmaxLayer(const int id, const string& name,Layer* prevLayer):Layer(id,name, prevLayer->m_tensorSize) {
    m_type = "SoftmaxLayer";
    addPreviousLayer(prevLayer);
}

SoftmaxLayer::~SoftmaxLayer(){

}

void SoftmaxLayer::initialize(const string& initialMethod){
    //null
}

void SoftmaxLayer::zeroParaGradient(){
    //null
}

void SoftmaxLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int nSoftmax = m_pYTensor->getDims()[0];// a vector's dimension to execute softmax
    const int N = X.getLength()/nSoftmax;  // the number of element vectors needing softmax
    for (int j=0; j<N; ++j){
        float sumExpX = 1e-8;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(X(i*N+j));
        }
        for (int i=0; i< nSoftmax; ++i){
            Y(i*N+j) = exp(X(i*N+j))/sumExpX;
        }
    }
}

void SoftmaxLayer::backward(bool computeW, bool computeX){
    if (!computeX) return;
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int nSoftmax = m_pdYTensor->getDims()[0];// a vector's dimension to execute softmax
    const int N = X.getLength()/nSoftmax;  // the number of element vectors needing softmax
    for (int j=0; j<N; ++j){
        float sumExpX = 1e-8;
        for (int i=0; i< nSoftmax; ++i){
            sumExpX += exp(X(i*N+j));
        }
        float sumExpX2 = sumExpX*sumExpX;

        // \sum(dL/dy_j*exp(x_j)
        float dyDotExpX = 0;
        for(int i=0; i< nSoftmax; ++i){
            dyDotExpX += dY(i*N+j)*exp(X(i*N+j));
        }

        for(int i=0; i< nSoftmax; ++i){
            dX(i*N+j) += exp(X(i*N+j))*(dY(i*N+j)*sumExpX-dyDotExpX)/sumExpX2;
        }

    }
}
void SoftmaxLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //Null
}

int  SoftmaxLayer::getNumParameters(){
    return 0;
}

void SoftmaxLayer::save(const string &netDir) {
//null
}

void SoftmaxLayer::load(const string &netDir) {
//null
}

void SoftmaxLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void SoftmaxLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
