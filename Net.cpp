//
// Created by Sheen156 on 6/5/2018.
//

#include "Net.h"
#include "InputLayer.h"
#include "FCLayer.h"
#include "ReLU.h"
#include "LossLayer.h"
#include <iostream>

Net::Net(){
   m_layers.clear();
}
Net::~Net(){
   while (0 != m_layers.size()){
      Layer* pLayer = m_layers.back();
      m_layers.pop_back();
      delete pLayer;
   }
}

void Net::setBatchSize(const int batchSize){
   m_batchSize = batchSize;
}

void Net::setLearningRate(const float learningRate){
   m_learningRate = learningRate;
}

void Net::forwardPropagate(){

}
void Net::backwardPropagate(){

}

void Net::addLayer(Layer* layer){
     m_layers.push_back(layer);
}

void Net::sgd(const float lr){

}

//Notes: this layerWidthVector does not include LossLayer,  and ReLU dose not count as a single layer
void Net::buildNet(const vector<long> layerWidthVector){
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
      }
   }
   LossLayer* lossLayer = new LossLayer(m_layers.back());
   addLayer(lossLayer);
}

void Net::initilize(){
   int i=0;
   for(list<Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
      if (0 == i) continue;
      else (*iter)->initialize("Xavier");
      ++i;
   }

}
void Net::train(const int nIteration)
{



}
