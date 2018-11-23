//
// Created by Hui Xie on 8/4/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <MergerLayer.h>

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

void MergerLayer::backward(bool computeW, bool computeX){
    if (computeX){
        for(list<Layer*>::iterator it= m_prevLayers.begin(); it != m_prevLayers.end();++it){
            *((*it)->m_pdYTensor) += *m_pdYTensor;
        }
    }
}

void MergerLayer::updateParameters(const float lr, const string& method, const int batchSize){
  //null
}

void MergerLayer::addPreviousLayer(Layer* prevLayer)
{
    if (nullptr != prevLayer){
        if (isLayerInList(prevLayer)){
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

void MergerLayer::delPreviousLayer(Layer* prevLayer){
    if (nullptr != prevLayer  &&  isLayerInList(prevLayer) ){
        delLayerFromList(prevLayer);
    }
}

bool MergerLayer::isLayerInList(const Layer* layer){
    for(list<Layer*>::const_iterator iter= m_prevLayers.begin(); iter!= m_prevLayers.end(); ++iter){
        if (layer == *iter) return true;
    }
    return false;
}

void MergerLayer::delLayerFromList(const Layer* layer){
    for(list<Layer*>::iterator iter= m_prevLayers.begin(); iter!= m_prevLayers.end(); ++iter){
        if (layer == *iter){
          iter = m_prevLayers.erase(iter);
          --iter;
        }
    }
}

long  MergerLayer::getNumParameters(){
    return 0;
}

void MergerLayer::save(const string &netDir) {
//null
}

void MergerLayer::load(const string &netDir) {
//null
}

void MergerLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    string prevLayersStr="";
    int N = m_prevLayers.size();
    list<Layer*>::const_iterator iterTemp;
    for (list<Layer*>::const_iterator iter = m_prevLayers.begin(); iter != m_prevLayers.end(); ++iter){
        iterTemp = iter;
        prevLayersStr += to_string((*iter)->m_id) + ((++iterTemp == m_prevLayers.end())?"":"_");
    }
    fprintf(pFile, "%d, %s, %s, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), prevLayersStr.c_str(),
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void MergerLayer::printStruct(const int layerIndex) {
    int size = m_prevLayers.size();
    string branchList = "{";
    for (list<Layer*> ::const_iterator it = m_prevLayers.begin(); it!= m_prevLayers.end(); ++it){
        branchList += (*it)->m_name + " ";
    }
    branchList += "}";

    printf("Layer%03d, Name=%s: (%s, id=%d): PreviousBranchList=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  branchList.c_str(), vector2Str(m_tensorSize).c_str());
}
