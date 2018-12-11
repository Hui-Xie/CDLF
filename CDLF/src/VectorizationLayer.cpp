//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <VectorizationLayer.h>

VectorizationLayer::VectorizationLayer(const int id, const string& name,Layer* prevLayer)
   : ReshapeLayer(id,name, prevLayer, {prevLayer->m_pYTensor->getLength(),1}){
    m_type = "VectorizationLayer";
    addPreviousLayer(prevLayer);
}

VectorizationLayer::~VectorizationLayer(){

}
