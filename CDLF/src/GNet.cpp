//
// Created by Hui Xie on 9/18/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "GNet.h"


GNet::GNet(const string& name, const string& saveDir): FeedForwardNet(name, saveDir){
    m_pGxLayer = nullptr;
    m_pInputXLayer = nullptr;
    m_pLossLayer = nullptr;
}

GNet::~GNet(){

}