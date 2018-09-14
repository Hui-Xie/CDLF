//
// Created by Hui Xie on 7/19/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "BiasLayer.h"
#include "statisTool.h"


BiasLayer::BiasLayer(const int id, const string& name, Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
   m_type = "BiasLayer";
   addPreviousLayer(prevLayer);
   m_pBTensor =   new Tensor<float>(prevLayer->m_tensorSize);
   m_pdBTensor =  new Tensor<float>(prevLayer->m_tensorSize);
}

BiasLayer::~BiasLayer() {
  if (nullptr != m_pBTensor) {
      delete m_pBTensor;
      m_pBTensor = nullptr;
  }
  if (nullptr != m_pdBTensor) {
      delete m_pdBTensor;
      m_pdBTensor = nullptr;
  }
}


void BiasLayer::initialize(const string& initialMethod){
  long N = m_pBTensor->getLength();
  generateGaussian(m_pBTensor, 0, sqrt(1.0/N));
}

void BiasLayer::zeroParaGradient(){
    if (nullptr != m_pdBTensor) m_pdBTensor->zeroInitialize();
}

//Y = X + b
void BiasLayer::forward(){
  *m_pYTensor = *m_prevLayer->m_pYTensor + *m_pBTensor;
}

// dL/dX = dL/dY    Where L is Loss
// dL/db = dL/dY
void BiasLayer::backward(){
    *m_pdBTensor += *m_pdYTensor;
    *(m_prevLayer->m_pdYTensor) += *m_pdYTensor;
}

void BiasLayer::updateParameters(const float lr, const string& method, const int batchSize){
    if ("sgd" == method){
        *m_pBTensor -=  (*m_pdBTensor)*(lr/batchSize);
    }
}