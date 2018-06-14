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

Net::Net(){
   m_layers.clear();
   m_learningRate = 0.01;
   m_lossTolerance = 0.02;
   m_maxIteration = 1000;
   m_judgeLoss = true;
}
Net::~Net(){
   while (0 != m_layers.size()){
      Layer* pLayer = m_layers.back();
      m_layers.pop_back();
      delete pLayer;
   }
}

void Net::setLearningRate(const float learningRate){
   m_learningRate = learningRate;
}

void Net::setLossTolerance(const float tolerance){
   m_lossTolerance = tolerance;
}

void Net::setMaxItration(const int maxIteration){
    m_maxIteration = maxIteration;
}

void Net::setJudgeLoss(const bool judgeLoss){
    m_judgeLoss = judgeLoss;
}

void Net::forwardPropagate(){
   for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
      (*iter)->forward();
   }
}
void Net::backwardPropagate(){
   for (list<Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
      (*rit)->backward();
   }
}

void Net::addLayer(Layer* layer){
     m_layers.push_back(layer);
}

void Net::sgd(const float lr){
    for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        (*iter)->updateParameters(lr, "sgd");
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
   InputLayer* inputLayer = new InputLayer(layerWidthVector.at(0));
   addLayer(inputLayer);
   for(int i =1; i< nLayers; ++i){
      FCLayer* fcLayer = new FCLayer(layerWidthVector.at(i),m_layers.back());
      addLayer(fcLayer);
      if (i != nLayers -1){
         ReLU* reLU = new ReLU(m_layers.back());
         addLayer(reLU);
         NormalizationLayer* normalLayer = new NormalizationLayer(m_layers.back());
         addLayer(normalLayer);
      }
   }
   lossLayer->setPreviousLayer(m_layers.back());
   addLayer(lossLayer);
}

void Net::initialize(){
   for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       (*iter)->initialize("Xavier");
    }
}
void Net::train()
{
   int nIter = 0;
   const int nLayers = m_layers.size();
   InputLayer* inputLayer = (InputLayer*)m_layers.front();
   LossLayer* lossLayer = (LossLayer* )m_layers.back();
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
    cout<<"Iteration: " << nIter << "  "
        <<"Output Result: "<< trans(*(lossLayer->m_prevLayerPointer->m_pYVector)) << "  "
        <<"Loss: "<< lossLayer->getLoss()<< endl;
    cout<<endl;
}

void Net::printLayersY(){
    for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        (*iter)->printY();
    }
}

void Net::printLayersDY(){
    for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        (*iter)->printDY();
    }
}

void Net::printLayersWdW(){
    for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if ("FullyConnected" == (*iter)->m_type){
            ((FCLayer*)(*iter))->printWandBVector();
            ((FCLayer*)(*iter))->printdWanddBVector();
        }
     }
}
