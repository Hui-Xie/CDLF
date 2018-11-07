//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "LossLayer.h"
#include <cmath>
#include <iostream>
#include <LossLayer.h>

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

void LossLayer::setGroundTruth( const Tensor<unsigned  char>& groundTruth){
    if (nullptr == m_pGroundTruth){
        m_pGroundTruth = new Tensor<float> (groundTruth.getDims());
    }
    long N = m_pGroundTruth->getLength();
    for (long i=0; i<N; ++i){
        m_pGroundTruth->e(i) = (float) groundTruth.e(i);
    }
}

long LossLayer::getNumParameters(){
    return 0;
}

void LossLayer::save(const string &netDir) {
   //null
}

void LossLayer::load(const string &netDir) {
  //null
}

void LossLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, previousLayerIDs, outputTensorSize, filterSize, numFilter, FilterStride, startPosition, \r\n";
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}


