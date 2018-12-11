//
// Created by Hui Xie on 12/11/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <ReshapeLayer.h>


ReshapeLayer::ReshapeLayer(const int id, const string& name,Layer* prevLayer, const vector<long>& outputSize)
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
    const long N = X.getLength();
    Y.copyDataFrom(X.getData(), N*sizeof(float));
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

void ReshapeLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

long  ReshapeLayer::getNumParameters(){
    return 0;
}

void ReshapeLayer::save(const string &netDir) {
//null
}

void ReshapeLayer::load(const string &netDir) {
//null
}

void ReshapeLayer::saveStructLine(FILE *pFile) {
//const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void ReshapeLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s: (%s, id=%d): PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
