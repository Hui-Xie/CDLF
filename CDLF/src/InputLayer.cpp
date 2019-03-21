//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <InputLayer.h>

#include "InputLayer.h"
#include "statisTool.h"
#include "Tools.h"

InputLayer::InputLayer(const int id, const string& name, const vector<int>& tensorSize): Layer(id, name, tensorSize){
    m_type = "InputLayer";

}

InputLayer::~InputLayer(){

}

void InputLayer::initialize(const string& initialMethod){
    //null
}

void InputLayer::zeroParaGradient(){
   //null
}

void InputLayer::forward(){
    //null
}
void InputLayer::backward(bool computeW, bool computeX){
    //null
}

void InputLayer::initializeLRs(const float lr) {
    //null
}

void InputLayer::updateLRs(const float deltaLoss) {
    //null
}

void InputLayer::updateParameters(Optimizer* pOptimizer) {
    //null
}

int InputLayer::getNumParameters() {
    return 0;
}

void InputLayer::save(const string &netDir) {
   //null
}

void InputLayer::load(const string &netDir) {
  //null
}

void InputLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), 0, vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void InputLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(), vector2Str(m_tensorSize).c_str());
}

void InputLayer::averageParaGradient(const int batchSize) {

}


