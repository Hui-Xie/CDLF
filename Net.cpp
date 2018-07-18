//
// Created by Sheen156 on 6/5/2018.
//

#include "Net.h"
#include "InputLayer.h"
#include "FCLayer.h"
#include "ReLU.h"
#include "LossLayer.h"
#include "NormalizationLayer.h"
#include <iostream>
#include <cmath> //for isinf()

Net::Net(){
   m_layers.clear();
   m_learningRate = 0.01;
   m_lossTolerance = 0.02;
   m_maxIteration = 1000;
   m_judgeLoss = true;
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

void Net::setMaxIteration(const int maxIteration){
    m_maxIteration = maxIteration;
}

void Net::setJudgeLoss(const bool judgeLoss){
    m_judgeLoss = judgeLoss;
}

void Net::forwardPropagate(){
   for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
      iter->second->forward();
   }
}
void Net::backwardPropagate(){
   for (map<int, Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
      rit->second->backward();
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

void Net::sgd(const float lr){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->updateParameters(lr, "sgd");
    }
}

//Notes: this layerWidthVector does not include LossLayer,  and ReLU and NormalizationLayer do not count as a single layer
// Normalization layer generally put after ReLU
void Net::buildNet(const vector<long> layerWidthVector, Layer* lossLayer){
   int nLayers = layerWidthVector.size();
   if (0 == nLayers) {
      cout<<"Net has at least one layer."<<endl;
      return;
   }
   int layerID = 1;
   InputLayer* inputLayer = new InputLayer(layerID++, "Input Layer",{layerWidthVector.at(0),1});
   addLayer(inputLayer);
   for(int i =1; i< nLayers; ++i){
      FCLayer* fcLayer = new FCLayer(layerID++, "FCLayer"+to_string(i), {layerWidthVector.at(i),1},m_layers.rbegin()->second);
      addLayer(fcLayer);
      if (i != nLayers -1){
         ReLU* reLU = new ReLU(layerID++, "ReLU"+to_string(i), m_layers.rbegin()->second);
         addLayer(reLU);
         NormalizationLayer* normalLayer = new NormalizationLayer(layerID++, "NormLayer"+to_string(i),m_layers.rbegin()->second);
         addLayer(normalLayer);
      }
   }
   lossLayer->addPreviousLayer(m_layers.rbegin()->second);
   addLayer(lossLayer);
}

void Net::initialize(){
   for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->initialize("Xavier");
    }
}
void Net::train()
{
   int nIter = 0;
   const int nLayers = m_layers.size();
   InputLayer* inputLayer = (InputLayer*)m_layers.begin()->second;
   LossLayer* lossLayer = (LossLayer* )m_layers.rbegin()->second;
   while(nIter < m_maxIteration && (m_judgeLoss ? lossLayer->getLoss()> m_lossTolerance: true))
   {
      inputLayer->initialize("Gaussian");
      forwardPropagate();
      backwardPropagate();

      float loss = lossLayer->getLoss();
      if (isinf(loss)) break;

      //debug:
      //printLayersY();
      //printLayersDY();
      //printLayersWdW();
      //cout<<"===================An iteration ============="<<endl;
      //end of debug
      sgd(m_learningRate);
      printIteration(lossLayer, nIter);
      ++nIter;
   }
   lossLayer->printGroundTruth();
}

void Net::printIteration(LossLayer* lossLayer, const int nIter){
    cout<<"Iteration: " << nIter << "  "  <<"Output Result: "<<endl;
    lossLayer->m_prevLayers.front()->m_pYTensor->transpose().printElements();
    cout<<"Loss: "<< lossLayer->getLoss()<< endl;
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
