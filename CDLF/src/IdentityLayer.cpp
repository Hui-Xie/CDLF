//
// Created by Hui Xie on 8/3/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <IdentityLayer.h>

#include "IdentityLayer.h"

IdentityLayer::IdentityLayer(const int id, const string& name,Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "IdentityLayer";
    addPreviousLayer(prevLayer);
}

IdentityLayer::~IdentityLayer(){

}

// Y = X
// dL/dx = dL/dy * dy/dx = dL/dy
void IdentityLayer::forward(){
    *m_pYTensor = *m_prevLayer->m_pYTensor;
}
void IdentityLayer::backward(bool computeW){
   *m_prevLayer->m_pdYTensor += *m_pdYTensor;
}
void IdentityLayer::initialize(const string& initialMethod){
    //null
}

void IdentityLayer::zeroParaGradient(){
    //null
}

void IdentityLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

long IdentityLayer::getNumParameters(){
    return 0;
}

void IdentityLayer::save(const string &netDir) {

}

void IdentityLayer::load(const string &netDir) {

}

void IdentityLayer::saveArchitectLine(FILE *pFile) {

}
