//
// Created by Hui Xie on 9/15/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "GAN.h"


GAN::GAN(const string& name, GNet* pGNet, DNet* pDNet){
    m_name = name;
    m_pGNet = pGNet;
    m_pDNet = pDNet;

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
  m_pDNet->backwardPropagate();
  copyGxGradientFromDtoG();
  m_pGNet->backwardPropagate();
}

void GAN::backwardD(){
  m_pDNet->backwardPropagate();
}

void GAN::switchDToGT(){
   m_pDNet->m_pMerger->delPreviousLayer(m_pDNet->m_pGxLayer);
   m_pDNet->m_pMerger->addPreviousLayer(m_pDNet->m_pGTLayer);
}

void GAN::switchDToGx(){
    m_pDNet->m_pMerger->delPreviousLayer(m_pDNet->m_pGTLayer);
    m_pDNet->m_pMerger->addPreviousLayer(m_pDNet->m_pGxLayer);
}

void GAN::copyGxYFromGtoD(){
    *m_pDNet->m_pGxLayer->m_pYTensor = *m_pGNet->m_pGxLayer->m_pYTensor;

}

void GAN::copyGxGradientFromDtoG(){
    *m_pGNet->m_pGxLayer->m_pdYTensor = *m_pDNet->m_pGxLayer->m_pdYTensor;
}