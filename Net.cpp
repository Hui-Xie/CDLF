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

void Net::setMaxIteration(const long maxIteration){
    m_maxIteration = maxIteration;
}

void Net::setJudgeLoss(const bool judgeLoss){
    m_judgeLoss = judgeLoss;
}

void Net::setBatchSize(const int batchSize){
    m_batchSize = batchSize;
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

//Notes: this layerWidthVector does not include LossLayer,  and ReLU and NormalizationLayer do not count as a single layer
// Normalization layer generally put after ReLU
void Net::buildFullConnectedNet(const vector<long> layerWidthVector, Layer* lossLayer){
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
   long nIter = 0;
   InputLayer* inputLayer = (InputLayer*)m_layers.begin()->second;
   LossLayer* lossLayer = (LossLayer* )m_layers.rbegin()->second;
   long numBatch =  m_maxIteration / m_batchSize;
   if (0 !=  m_maxIteration % m_batchSize){
       numBatch += 1;
   }

   long nBatch = 0;
   while(nBatch < numBatch)
   {
      if (m_judgeLoss && lossLayer->getLoss()< m_lossTolerance){
         break;
       }
      if (isinf(lossLayer->getLoss())) break;

      zeroParaGradient();
      int i=0;
      for(i=0; i< m_batchSize && nIter < m_maxIteration; ++i){
          inputLayer->initialize("Gaussian");
          forwardPropagate();
          backwardPropagate();
          ++nIter;
      }
      sgd(m_learningRate,i);

      //debug:
      //printLayersY();
      //printLayersDY();
      //printLayersWdW();
      //cout<<"===================An iteration ============="<<endl;
      //end of debug

      printIteration(lossLayer, nIter);
      ++nBatch;
   }
   lossLayer->printGroundTruth();
}

void Net::printIteration(LossLayer* lossLayer, const int nIter){
    cout<<"Iteration: " << nIter << "  "  <<"Output Result: "<<endl;
    lossLayer->m_prevLayers.front()->m_pYTensor->transpose().printElements();
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
