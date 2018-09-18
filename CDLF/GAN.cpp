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

void GAN::sgdG(){

}

void GAN::sgdD(){

}

void GAN::switchToGT(){

}

void GAN::switchToGx(){

}

void GAN::copyGxYFromGtoD(){

}

void GAN::copyGxGradientFromDtoG(){

}



