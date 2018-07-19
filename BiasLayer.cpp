//
// Created by Sheen156 on 7/19/2018.
//

#include "BiasLayer.h"
#include "statisTool.h"


BiasLayer::BiasLayer(const int id, const string& name, Layer* prevLayer): Layer(id,name, prevLayer->m_tensorSize){
   m_type = "BiasLayer";
   addPreviousLayer(prevLayer);
   m_pBTensor =   new Tensor<float>(prevLayer->m_tensorSize);
   m_pdBTensor =  new Tensor<float>(prevLayer->m_tensorSize);
}

BiasLayer::~BiasLayer() {
  if (nullptr != m_pBTensor) delete m_pBTensor;
  if (nullptr != m_pdBTensor) delete m_pdBTensor;
}


void BiasLayer::initialize(const string& initialMethod){
  long N = m_pBTensor->getLength();
  generateGaussian(m_pBTensor, 0, 1.0/N);
}

//Y = X + b
void BiasLayer::forward(){
  *m_pYTensor = *m_prevLayers.front()->m_pYTensor + *m_pBTensor;
}

// dL/dX = dL/dY    Where L is Loss
// dL/db = dL/dY
void BiasLayer::backward(){
    *m_pdBTensor = *m_pdYTensor;
    *m_prevLayers.front()->m_pdYTensor = *m_pdYTensor;
}

void BiasLayer::updateParameters(const float lr, const string& method){
    if ("sgd" == method){
        *m_pBTensor -=  (*m_pdBTensor)*lr;
    }
}