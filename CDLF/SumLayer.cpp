//
// Created by Hui Xie on 8/4/2018.
//

#include "SumLayer.h"

SumLayer::SumLayer(const int id, const string& name, const vector<long>& tensorSize): Layer(id, name, tensorSize)
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

void SumLayer::addPreviousLayer(Layer* prevLayer)
{
    if (nullptr != prevLayer){
        if (isLayerInList(prevLayer)){
            cout<<"Error: repeatedly add layer to SumLayer:"<<prevLayer->m_name<<endl;
            return;
        }

        m_prevLayers.push_back(prevLayer);
    }
}

bool SumLayer::isLayerInList(const Layer* layer){
    for(list<Layer*>::const_iterator iter= m_prevLayers.begin(); iter!= m_prevLayers.end(); ++iter){
        if (layer == *iter) return true;
    }
    return false;
}