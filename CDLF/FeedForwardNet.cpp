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

}

FeedForwardNet::~FeedForwardNet() {
    for (map<int, Layer *>::iterator it = m_layers.begin(); it != m_layers.end(); ++it) {
        delete it->second;
        it->second = nullptr;
    }
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




void FeedForwardNet::sgd(const float lr, const int batchSize){
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->updateParameters(lr, "sgd", batchSize);
    }
}

