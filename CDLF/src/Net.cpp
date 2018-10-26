//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "Net.h"
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "GPUAttr.h"

Net::Net(const string& name){
    m_name = name;
    m_layers.clear();
    m_learningRate = 0.001;
    m_lossTolerance = 0.02;
    m_judgeLoss = true;
    m_batchSize = 1;
}

Net::~Net() {
    for (map<int, Layer *>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        if (nullptr != it->second){
            delete it->second;
            it->second = nullptr;
        }
    }
    m_layers.clear();
}

void Net::setLearningRate(const float learningRate){
    m_learningRate = learningRate;
}

void Net::setLossTolerance(const float tolerance){
    m_lossTolerance = tolerance;
}

void Net::setJudgeLoss(const bool judgeLoss){
    m_judgeLoss = judgeLoss;
}

void Net::setBatchSize(const int batchSize){
    m_batchSize = batchSize;
}

void Net::setEpoch(const long epoch){
    m_epoch = epoch;
}

string Net::getName(){
   return m_name;
}

float Net::getLearningRate(){
    return m_learningRate;
}
float Net::getLossTolerance(){
    return m_lossTolerance;
}

bool Net::getJudgeLoss(){
    return m_judgeLoss;
}
int  Net::getBatchSize(){
    return m_batchSize;
}

long  Net::getEpoch(){
    return m_epoch;
}

map<int, Layer*> Net::getLayersMap(){
    return m_layers;
}

long Net::getNumParameters(){
    long num = 0;
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        num +=  iter->second->getNumParameters();
    }
    return num;
}

void Net::zeroParaGradient(){
    for (map<int, Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
        rit->second->zeroParaGradient();
    }
}

void Net::addLayer(Layer* layer){
    if (nullptr == layer) return;
    if (0 == m_layers.count(layer->m_id) && !layerExist(layer)){
        m_layers[layer->m_id] = layer;
    }
    else{
        cout<<"Error: repeated layer ID or name in added layer."<<endl;
        cout<<"\t layer ID: "<< layer->m_id<<endl;
        cout<<"\t already existed layer: "<<m_layers[layer->m_id]->m_name<<endl;
        cout<<"\t new adding layer: "<<layer->m_name<<endl;
    }
}

Layer* Net::getLayer(const int ID){
    return m_layers.at(ID);
}


InputLayer* Net::getInputLayer(){
    return (InputLayer*) m_layers.begin()->second;
}

Layer*  Net::getFirstLayer(){
    return  m_layers.begin()->second;
}
Layer* Net::getFinalLayer(){
    return m_layers.rbegin()->second;
}

void Net::initialize(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->initialize("Xavier");
    }
}

void Net::printIteration(LossLayer* lossLayer, const int nIter){
    cout<<"Iteration: " << nIter << "  "  <<"Output Result: "<<endl;
    long N = lossLayer->m_prevLayer->m_pYTensor->getLength();
    lossLayer->m_prevLayer->m_pYTensor->reshape({1,N}).printElements();
    if (nullptr != lossLayer->m_pGroundTruth){
        cout<<"GrounTruth: "<<endl;
        lossLayer->m_pGroundTruth->reshape({1,N}).printElements();
    }
    cout<<"Loss: "<< lossLayer->lossCompute()<< endl;
    cout<<endl;
}

void Net::printLayersY(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->printY();
    }
}

void Net::printLayersDY(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->printDY();
    }
}


void Net::printArchitecture(){
    cout<<endl<<"========== Network Architecture of "<<m_name<<" ============="<<endl;
    cout<<"======================================================"<<endl;
    int i=1;
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        cout<<"Layer_"<<i++<<" ("<<iter->second->m_type<<", id="<<std::to_string(iter->second->m_id)<<"): "<<iter->second->m_name<<" : ";
        if (nullptr != iter->second->m_prevLayer){
            cout<<"PreviousLayer = "<<iter->second->m_prevLayer->m_name<<"; ";
        }
        if ( "ConvolutionLayer"==iter->second->m_type){
            cout<<"FilterSize = "<<vector2Str(((ConvolutionLayer*)iter->second)->m_filterSize)<<"; "<<"NumOfFilter = "<<((ConvolutionLayer*)iter->second)->m_numFilters<<"; ";
        }
        if ( "MaxPoolingLayer"==iter->second->m_type){
            cout<<"FilterSize = "<<vector2Str(((MaxPoolingLayer*)iter->second)->m_filterSize)<<"; ";
        }
        cout<<"OutputSize = "<<vector2Str(iter->second->m_tensorSize)<<"; "<<endl;
    }

    cout<<"This network has total "<<getNumParameters()<<" learning parameters. "<<endl;
    cout<<"=========== End of Network Architecture =============="<<endl;
}

bool Net::layerExist(const Layer* layer){
    for(map<int, Layer*>::const_iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if (layer->m_name == iter->second->m_name || layer == iter->second ){
            cout<<"Error: "<<layer->m_name<<" has already been in the previous added layer."<<endl;
            return true;
        }
    }
    return false;
}
