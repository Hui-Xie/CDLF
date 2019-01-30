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

void LossLayer::printGroundTruth() {
    if (nullptr != m_pGroundTruth){
        m_pGroundTruth->print();
    }
}

float LossLayer::getLoss(){
    return m_loss;
}

void LossLayer::forward(){
    if(nullptr != m_pGroundTruth){
        lossCompute();
    }
}
void LossLayer::backward(bool computeW, bool computeX){
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



int LossLayer::getNumParameters(){
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

void LossLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(), m_prevLayer->m_name.c_str());
}


float LossLayer::diceCoefficient(const float threshold) {
    // just compute the target volume, without considering the background
    const Tensor<float>* pPredict = m_prevLayer->m_pYTensor;
    const Tensor<float>* pGT =  m_pGroundTruth;
    const int N = pPredict->getLength();
    int nPredict = 0;
    int nGT = 0;
    int nIntersection = 0;
    for (int i=0; i< N; ++i)
    {
        nIntersection += (pPredict->e(i) >= threshold && pGT->e(i) >= threshold)? 1: 0;
        nPredict += (pPredict->e(i) >= threshold)? 1:0;
        nGT += (pGT->e(i) >= threshold)? 1:0;
    }
    return nIntersection*2.0/(nPredict+nGT);
}


// TruePositiveRate = recall= sensitivity = TP/(TP+FN)
float LossLayer::getTPR(const float threshold){
    const Tensor<float>* pPredict = m_prevLayer->m_pYTensor;
    const Tensor<float>* pGT =  m_pGroundTruth;
    const int N = pPredict->getLength();
    int nTP = 0; // True Positive
    int nP = 0;// nP = nTP + nFP
    for (int i=0; i< N; ++i)
    {
        if (pGT->e(i) >= threshold)
        {
            ++nP;
            if (pPredict->e(i) >= threshold ){
                ++nTP;
            }
        }
    }
    return nTP*1.0/nP;
}


