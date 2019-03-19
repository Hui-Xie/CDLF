//
// Created by Hui Xie on 12/11/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <ReshapeLayer.h>


ReshapeLayer::ReshapeLayer(const int id, const string& name,Layer* prevLayer, const vector<int>& outputSize)
        : Layer(id,name, outputSize){
    m_type = "ReshapeLayer";
    addPreviousLayer(prevLayer);

    if (length(outputSize) != prevLayer->m_pYTensor->getLength()){
        cout<<"Error: reshape can not change the length of previous layer."<<endl;
    }
}

ReshapeLayer::~ReshapeLayer(){

}

void ReshapeLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int N = X.getLength();
    Y.copyDataFrom(X.getData(), N);
}
void ReshapeLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        dX += dY.reshape(dX.getDims());
    }
}
void ReshapeLayer::initialize(const string& initialMethod){
    //null
}

void ReshapeLayer::zeroParaGradient(){
    //null
}



int  ReshapeLayer::getNumParameters(){
    return 0;
}

void ReshapeLayer::save(const string &netDir) {
//null
}

void ReshapeLayer::load(const string &netDir) {
//null
}

void ReshapeLayer::saveStructLine(FILE *pFile) {
//const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void ReshapeLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}

void ReshapeLayer::initializeLRs(const float lr) {

}

void ReshapeLayer::updateLRs(const float deltaLoss) {

}

void ReshapeLayer::updateParameters(const string& method, Optimizer* pOptimizer) {

}

void ReshapeLayer::averageParaGradient(const int batchSize) {

}
