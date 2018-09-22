//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "DNet.h"
#include "CrossEntropyLoss.h"


DNet::DNet(const string& name):FeedForwardNet(name){
    m_pGTLayer = nullptr;
    m_pGxLayer = nullptr;
    m_pInputXLayer = nullptr;
    m_pMerger = nullptr;
    m_pLossLayer = nullptr;
}

DNet::~DNet(){

}
/* [0,1]' indicate alpha = true;
 * [1,0]' indicate alpha = false;
 * */

void DNet::setAlphaGroundTruth(bool alpha){
    Tensor<float>* pGT = m_pLossLayer->m_pGroundTruth;
    pGT->zeroInitialize();
    if (alpha) pGT->e(1) =1;
    else pGT->e(0) =1;
}