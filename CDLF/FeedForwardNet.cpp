//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "FeedForwardNet.h"
#include "InputLayer.h"
#include "FCLayer.h"
#include "ReLU.h"
#include "LossLayer.h"
#include "NormalizationLayer.h"
#include <iostream>
#include <cmath> //for isinf()
#include "statisTool.h"
#include "ConvolutionLayer.h"

FeedForwardNet::FeedForwardNet(){
   m_layers.clear();
   m_learningRate = 0.01;
   m_lossTolerance = 0.02;
   m_judgeLoss = true;
   m_batchSize = 1;
}

FeedForwardNet::~FeedForwardNet() {
    for (map<int, Layer *>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        delete it->second;
        it->second = nullptr;
    }
}

void FeedForwardNet::setLearningRate(const float learningRate){
   m_learningRate = learningRate;
}

void FeedForwardNet::setLossTolerance(const float tolerance){
   m_lossTolerance = tolerance;
}

void FeedForwardNet::setJudgeLoss(const bool judgeLoss){
    m_judgeLoss = judgeLoss;
}

void FeedForwardNet::setBatchSize(const int batchSize){
    m_batchSize = batchSize;
}

void FeedForwardNet::setEpoch(const long epoch){
    m_epoch = epoch;
}

float FeedForwardNet::getLearningRate(){
    return m_learningRate;
}
float FeedForwardNet::getLossTolerance(){
   return m_lossTolerance;
}

bool FeedForwardNet::getJudgeLoss(){
  return m_judgeLoss;
}
int  FeedForwardNet::getBatchSize(){
  return m_batchSize;
}

long  FeedForwardNet::getEpoch(){
  return m_epoch;
}

map<int, Layer*> FeedForwardNet::getLayersMap(){
    return m_layers;
}

void FeedForwardNet::forwardPropagate(){
    // Input Layer do not need zeroYTensor;
    for(map<int, Layer*>::iterator iter = ++(m_layers.begin()); iter != m_layers.end(); ++iter){
        iter->second->zeroYTensor();
    }
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->forward();
   }
}
void FeedForwardNet::backwardPropagate(){
   // first initialize all dy into zero.
   // this is a necessary step. as ConvolutionalLayer, MaxPoolLayer, ReLULayer all need this.
   // the previous layer of any layer may be a branch Layer, so when we compute m_pdYTensor, always use +=
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->zeroDYTensor();
   }
   for (map<int, Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
         rit->second->backward();
   }
}

void FeedForwardNet::zeroParaGradient(){
    for (map<int, Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
        rit->second->zeroParaGradient();
    }
}

void FeedForwardNet::addLayer(Layer* layer){
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

Layer* FeedForwardNet::getLayer(const int ID){
    return m_layers.at(ID);
}

void FeedForwardNet::sgd(const float lr, const int batchSize){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->updateParameters(lr, "sgd", batchSize);
    }
}

InputLayer* FeedForwardNet::getInputLayer(){
    return (InputLayer*) m_layers.begin()->second;
}

Layer*  FeedForwardNet::getFirstLayer(){
    return  m_layers.begin()->second;
}
Layer* FeedForwardNet::getFinalLayer(){
    return m_layers.rbegin()->second;
}

void FeedForwardNet::initialize(){
   for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->initialize("Xavier");
    }
}

void FeedForwardNet::printIteration(LossLayer* lossLayer, const int nIter){
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

void FeedForwardNet::printLayersY(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->printY();
    }
}

void FeedForwardNet::printLayersDY(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->printDY();
    }
}

void FeedForwardNet::printLayersWdW(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if ("FullyConnected" == iter->second->m_type){
            ((FCLayer*)(iter->second))->printWandBVector();
            ((FCLayer*)(iter->second))->printdWanddBVector();
        }
     }
}

void FeedForwardNet::printArchitecture(){
    cout<<endl<<"Network Architecture: "<<endl;
    int i=1;
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        cout<<"Layer_"<<i++<<" ("<<iter->second->m_type<<", id="<<std::to_string(iter->second->m_id)<<"): "<<iter->second->m_name<<" : "<<"OutputSize = "<<vector2Str(iter->second->m_tensorSize)<<"; ";
        if ( "ConvolutionLayer"==iter->second->m_type){
            cout<<"FilterSize = "<<vector2Str(((ConvolutionLayer*)iter->second)->m_filterSize)<<"; "<<"NumOfFilter = "<<((ConvolutionLayer*)iter->second)->m_numFilters<<"; ";
        }
        cout<<endl;
    }
    cout<<"=========== End of Network Architecture =============="<<endl;
}

bool FeedForwardNet::layerExist(const Layer* layer){
    for(map<int, Layer*>::const_iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if (layer->m_name == iter->second->m_name || layer == iter->second ){
            cout<<"Error: "<<layer->m_name<<" has already been in the previous added layer."<<endl;
            return true;
        }
    }
    return false;
}
