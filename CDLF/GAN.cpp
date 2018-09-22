//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "GAN.h"


GAN::GAN(const string& name, GNet* pGNet, DNet* pDNet){
    m_name = name;
    m_pGNet = pGNet;
    m_pDNet = pDNet;
    m_pStubLayer = nullptr;

}

GAN::~GAN(){

}

void GAN::forwardG(){
   m_pGNet->forwardPropagate();
   copyGxYFromGtoD();
   m_pDNet->forwardPropagate();
}

void GAN::forwardD(){
   m_pDNet->forwardPropagate();
}

void GAN::backwardG(){
  m_pDNet->backwardPropagate(false);
  copyGxGradientFromDtoG();
  m_pGNet->backwardPropagate(true);
}

void GAN::backwardD(){
  m_pDNet->backwardPropagate(true);
}

void GAN::switchDToGT(){
   m_pDNet->m_pMerger->delPreviousLayer(m_pDNet->m_pGxLayer);
   if (nullptr != m_pStubLayer) {
       m_pDNet->m_pMerger->delPreviousLayer(m_pStubLayer);
   }
   m_pDNet->m_pMerger->addPreviousLayer(m_pDNet->m_pGTLayer);
}

void GAN::switchDToGx(){
    m_pDNet->m_pMerger->delPreviousLayer(m_pDNet->m_pGTLayer);
    if (nullptr != m_pStubLayer) {
        m_pDNet->m_pMerger->delPreviousLayer(m_pStubLayer);
    }
    m_pDNet->m_pMerger->addPreviousLayer(m_pDNet->m_pGxLayer);
}

void GAN::switchDToStub(){
    m_pDNet->m_pMerger->delPreviousLayer(m_pDNet->m_pGTLayer);
    m_pDNet->m_pMerger->delPreviousLayer(m_pDNet->m_pGxLayer);
    m_pDNet->m_pMerger->addPreviousLayer(m_pStubLayer);
}

void GAN::setStubLayer(Layer* pStubLayer){
    m_pStubLayer = pStubLayer;
}

void GAN::copyGxYFromGtoD(){
    *m_pDNet->m_pGxLayer->m_pYTensor = *m_pGNet->m_pGxLayer->m_pYTensor;

}

void GAN::copyGxGradientFromDtoG(){
    *m_pGNet->m_pGxLayer->m_pdYTensor = *m_pDNet->m_pGxLayer->m_pdYTensor;
}