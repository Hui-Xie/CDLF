//
// Created by Hui Xie on 9/28/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <ScaleLayer.h>

#include "ScaleLayer.h"


ScaleLayer::ScaleLayer(const int id, const string& name, Layer* prevLayer,  const float k): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "ScaleLayer";
    addPreviousLayer(prevLayer);
    m_k = k;
    m_dk = 0;
}

ScaleLayer::~ScaleLayer() {
   //null
}


void ScaleLayer::initialize(const string& initialMethod){
    //null
}

void ScaleLayer::zeroParaGradient(){
    m_dk = 0;
}

//Y = k*X
void ScaleLayer::forward(){
    *m_pYTensor = *m_prevLayer->m_pYTensor * m_k;
}

/* y_i = k*x_i, where k is a learning scalar.
 * dL/dx_i = dL/dy_i *k
 * dL/dk = (dL/dy)' * x, where y and x are 1D vector form,prime symbol means transpose.
 *
 * */
void ScaleLayer::backward(bool computeW){
    if (computeW) {
        m_dk += m_pdYTensor->dotProduct(*m_prevLayer->m_pYTensor);
    }
    *(m_prevLayer->m_pdYTensor) += *m_pdYTensor * m_k;
}

void ScaleLayer::updateParameters(const float lr, const string& method, const int batchSize){
    if ("sgd" == method){
        m_k -=  m_dk*(lr/batchSize);
    }
}

long ScaleLayer::getNumParameters(){
    return 1;
}

void ScaleLayer::save(const string &netDir) {

}

void ScaleLayer::load(const string &netDir) {

}
