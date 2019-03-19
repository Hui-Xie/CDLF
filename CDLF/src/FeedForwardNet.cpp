//
// Created by Hui Xie on 6/5/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include "FeedForwardNet.h"
#include "InputLayer.h"
#include "FCLayer.h"
#include "ReLU.h"
#include "LossLayer.h"
#include "NormalizationLayer.h"
#include <iostream>
#include <cmath> //for isinf()
#include <FeedForwardNet.h>

#include "statisTool.h"
#include "ConvolutionLayer.h"

FeedForwardNet::FeedForwardNet(const string &saveDir) : Net(saveDir) {

}

FeedForwardNet::~FeedForwardNet() {

}

void FeedForwardNet::forwardPropagate() {
    //Input Layer do not need zeroYTensor;
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        if ("InputLayer" != iter->second->m_type) {
            iter->second->zeroYTensor();
        }
    }
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->forward();
#ifdef Dev_Debug
        printf("  ==> Foward layer of %s, finished at ", iter->second->m_name.c_str());
        printCurrentLocalTime();

#endif
    }
}

void FeedForwardNet::backwardPropagate(bool computeW) {
    // first initialize all dy into zero.
    // this is a necessary step. as ConvolutionalLayer, MaxPoolLayer, ReLULayer all need this.
    // the previous layer of any layer may be a branch Layer, so when we compute m_pdYTensor, always use +=
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->zeroDYTensor();
    }
    for (map<int, Layer *>::reverse_iterator rit = m_layers.rbegin(); rit != m_layers.rend(); ++rit) {
        if (rit->second->m_id > m_unlearningLayerID) {
            rit->second->backward(computeW, true);
#ifdef Dev_Debug
            printf("<== Backward layer of  %s, finished at ", rit->second->m_name.c_str());
            printCurrentLocalTime();
#endif
        }
        else if (rit->second->m_id == m_unlearningLayerID){
            rit->second->backward(computeW, false);
#ifdef Dev_Debug
            printf("<== Backward layer of  %s, finished at ", rit->second->m_name.c_str());
            printCurrentLocalTime();
#endif
        }
        else{
            continue;
        }
    }
}

void FeedForwardNet::optimize(const string& method){
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        if (iter->second->m_id >= m_unlearningLayerID) {
            iter->second->updateParameters(method,m_optimizer);
        }
    }
}

/*
void FeedForwardNet::initializeLRs(const float lr){
    setLearningRate(lr);
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->initializeLRs(lr);
    }
}

void FeedForwardNet::updateLearingRates(const float deltaLoss) {
    if (0 == deltaLoss) return;
    for (map<int, Layer *>::iterator iter = m_layers.begin(); iter != m_layers.end(); ++iter) {
        iter->second->updateLRs(deltaLoss);
    }
}

*/