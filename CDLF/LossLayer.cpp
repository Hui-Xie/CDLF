//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "LossLayer.h"
#include <cmath>
#include <iostream>
using namespace std;

LossLayer::LossLayer(const int id, const string& name, Layer *prevLayer) : Layer(id,name,{}){
    m_type = "LossLayer";
    m_loss = 1e+10;
    m_pGroundTruth = nullptr;
    addPreviousLayer(prevLayer);
}

LossLayer::~LossLayer(){
   if (nullptr != m_pGroundTruth){
       delete m_pGroundTruth;
       m_pGroundTruth = nullptr;
   }
}
float LossLayer::getLoss(){
    return m_loss;
}

void LossLayer::forward(){
    lossCompute();
}
void LossLayer::backward(bool computeW){
    gradientCompute();
}
void LossLayer::initialize(const string& initialMethod){
    //null
}

void LossLayer::zeroParaGradient(){
    //null
}

void LossLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

void LossLayer::setGroundTruth( const Tensor<float>& groundTruth){
    if (nullptr == m_pGroundTruth){
        m_pGroundTruth = new Tensor<float> (groundTruth.getDims());
    }
    * m_pGroundTruth = groundTruth;
}



