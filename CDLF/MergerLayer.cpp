//
// Created by Hui Xie on 8/4/2018.
//

#include "MergerLayer.h"

MergerLayer::MergerLayer(const int id, const string& name, const vector<long>& tensorSize): Layer(id, name, tensorSize)
{
    m_type = "MergerLayer";
}

MergerLayer::~MergerLayer()
{
  //null
}

void MergerLayer::initialize(const string& initialMethod){
  //null
}

void MergerLayer::zeroParaGradient(){
  //null
}

void MergerLayer::forward(){
  m_pYTensor->zeroInitialize();
  for(list<Layer*>::iterator it= m_prevLayers.begin(); it != m_prevLayers.end();++it){
      *m_pYTensor += *((*it)->m_pYTensor);
  }
}

void MergerLayer::backward(){
    for(list<Layer*>::iterator it= m_prevLayers.begin(); it != m_prevLayers.end();++it){
        *((*it)->m_pdYTensor) += *m_pdYTensor;
    }
}

void MergerLayer::updateParameters(const float lr, const string& method, const int batchSize){
  //null
}

void MergerLayer::addPreviousLayer(Layer* prevLayer)
{
    if (nullptr != prevLayer){
        if (isLayerInList(prevLayer)){
            cout<<"Error: repeatedly add layer to MergerLayer:"<<prevLayer->m_name<<endl;
            return;
        }
        if (!sameVector(m_tensorSize,prevLayer->m_tensorSize)){
            cout<<"Error: Incoming branches to MergerLayer should have same tensorSize"<<endl;
            cout<<"MergerLayer Tensor: "<< vector2Str(m_tensorSize)<<endl;
            cout<<"New prevlayer Tensor: "<< vector2Str(prevLayer->m_tensorSize)<<"  Layer Name: "<<prevLayer->m_name<<endl;
            return;
        }
        m_prevLayers.push_back(prevLayer);
    }
}

bool MergerLayer::isLayerInList(const Layer* layer){
    for(list<Layer*>::const_iterator iter= m_prevLayers.begin(); iter!= m_prevLayers.end(); ++iter){
        if (layer == *iter) return true;
    }
    return false;
}