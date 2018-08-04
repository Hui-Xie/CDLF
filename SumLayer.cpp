//
// Created by Sheen156 on 8/4/2018.
//

#include "SumLayer.h"

SumLayer::SumLayer(const int id, const string& name, const vector<int>& tensorSize): Layer(id, name, tensorSize)
{
    m_type = "SumLayer";
}

SumLayer::~SumLayer()
{
  //null
}

void SumLayer::initialize(const string& initialMethod){
  //null
}

void SumLayer::zeroParaGradient(){
  //null
}

void SumLayer::forward(){
  m_pYTensor->zeroInitialize();
  for(list<Layer*>::iterator it= m_prevLayers.begin(); it != m_prevLayers.end();++it){
      *m_pYTensor += *((*it)->m_pYTensor);
  }
}

void SumLayer::backward(){
    for(list<Layer*>::iterator it= m_prevLayers.begin(); it != m_prevLayers.end();++it){
        *((*it)->m_pdYTensor) += *m_pdYTensor;
    }
}

void SumLayer::updateParameters(const float lr, const string& method, const int batchSize){
  //null
}

void SumLayer::addPreviousLayer(Layer* prevLayer){
    if (nullptr != prevLayer){
        m_prevLayers.push_back(prevLayer);
        prevLayer->m_nextLayer = this;
    }
}