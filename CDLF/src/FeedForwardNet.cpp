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

FeedForwardNet::FeedForwardNet(const string& name):Net(name){

}

FeedForwardNet::~FeedForwardNet() {

}

void FeedForwardNet::forwardPropagate(){
    //Input Layer do not need zeroYTensor;
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        if ("InputLayer" != iter->second->m_type){
            iter->second->zeroYTensor();
        }
    }
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->forward();
       printf("  ==> Foward a layer: %s at ", iter->second->m_name.c_str());
       printCurrentLocalTime();
    }
}
void FeedForwardNet::backwardPropagate(bool computeW){
   // first initialize all dy into zero.
   // this is a necessary step. as ConvolutionalLayer, MaxPoolLayer, ReLULayer all need this.
   // the previous layer of any layer may be a branch Layer, so when we compute m_pdYTensor, always use +=
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
       iter->second->zeroDYTensor();
   }
   for (map<int, Layer*>::reverse_iterator rit=m_layers.rbegin(); rit!=m_layers.rend(); ++rit){
       rit->second->backward(computeW);
       printf("<== Backward a layer: %s at ", rit->second->m_name.c_str());
       printCurrentLocalTime();
   }
}

void FeedForwardNet::sgd(const float lr, const int batchSize){
    if (0 == batchSize) return;
    for(map<int, Layer*>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter){
        iter->second->updateParameters(lr, "sgd", batchSize);
    }
}