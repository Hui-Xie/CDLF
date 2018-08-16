//
// Created by Hui Xie on 6/5/2018.
//

#include "Net.h"
#include "InputLayer.h"
#include "FCLayer.h"
#include "ReLU.h"
#include "LossLayer.h"
#include "NormalizationLayer.h"
#include <iostream>
#include <cmath> //for isinf()
#include "statisTool.h"
#include "ConvolutionLayer.h"

Net::Net(){
   m_layers.clear();
   m_learningRate = 0.01;
   m_lossTolerance = 0.02;
   m_judgeLoss = true;
   m_batchSize = 1;
}

Net::~Net() {
    for (map<int, Layer *>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        delete it->second;
        it->second = nullptr;
    }
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

void Net::forwardPropagate(){
    // Input Layer do not need zeroYTensor;
    for(map<int, Layer*>::iterator iter = ++(m_layers.begin()); iter != m_layers.end(); ++iter){
        iter->second->zeroYTensor();
    }
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->forward();
   }
}
void Net::backwardPropagate(){
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

void Net::zeroParaGradient(){
    for (map<int, Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
        rit->second->zeroParaGradient();
    }
}

void Net::addLayer(Layer* layer){
     if (nullptr == layer) return;
     if (0 == m_layers.count(layer->m_id)){
         m_layers[layer->m_id] = layer;
     }
     else{
         cout<<"Error: repeated layer ID in adding layer."<<endl;
         cout<<"\t layer ID: "<< layer->m_id<<endl;
         cout<<"\t already existed layer: "<<m_layers[layer->m_id]->m_name<<endl;
         cout<<"\t new adding layer: "<<layer->m_name<<endl;
     }
}

void Net::sgd(const float lr, const int batchSize){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->updateParameters(lr, "sgd", batchSize);
    }
}

Layer* Net::getInputLayer(){
    return m_layers.begin()->second;
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
    lossLayer->m_prevLayer->m_pYTensor->transpose().printElements();
    if (nullptr != lossLayer->m_pGroundTruth){
        cout<<"GrounTruth: "<<endl;
        lossLayer->m_pGroundTruth->transpose().printElements();
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

void Net::printLayersWdW(){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if ("FullyConnected" == iter->second->m_type){
            ((FCLayer*)(iter->second))->printWandBVector();
            ((FCLayer*)(iter->second))->printdWanddBVector();
        }
     }
}

void Net::printArchitecture(){
    cout<<endl<<"Network Architecture: "<<endl;
    int i=1;
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        cout<<"Layer_"<<i++<<" ("<<iter->second->m_type<<") "<<": "<<iter->second->m_name<<": "<<"OutputSize = "<<vector2Str(iter->second->m_tensorSize)<<"; ";
        if ( "ConvolutionLayer"==iter->second->m_type){
            cout<<"FilterSize = "<<vector2Str(((ConvolutionLayer*)iter->second)->m_filterSize)<<"; "<<"NumOfFilter = "<<((ConvolutionLayer*)iter->second)->m_numFilters<<"; ";
        }
        cout<<endl;
    }
    cout<<"=========== End of Network Architecture =============="<<endl;
}
