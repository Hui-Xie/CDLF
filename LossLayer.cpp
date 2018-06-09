//
// Created by Sheen156 on 6/6/2018.
//

#include "LossLayer.h"
#include <cmath>

LossLayer::LossLayer(Layer* preLayer) : Layer(0){
    m_type = "LossLayer";
    m_prevLayerPointer = preLayer;
    m_prevLayerPointer->m_nextLayerPointer = this;
    m_loss = 0;
    m_avgLoss= 1e+10;
}

LossLayer::~LossLayer(){

}

void LossLayer::forward(){
    lossCompute();
}
void LossLayer::backward(){
    gradientCompute(m_avgLoss);
}
void LossLayer::initialize(const string& initialMethod){
    //null
}

void LossLayer::updateParameters(const float lr, const string& method) {
    //null
}

float LossLayer::lossCompute(){
    //use m_prevLayerPointer->m_pYVector,
    // An Example
    m_loss = 0;
    long N = m_prevLayerPointer->m_pYVector->size();
    DynamicVector<float> & prevY = *(m_prevLayerPointer->m_pYVector);
    for (long i=0; i< N ;++i){
        m_loss += pow( prevY[i] - i , 2);
    }
    return m_loss;
}

// f= \sum (x_i-i)^2
// Loss = (f-0)^2
// dL/dx_i = dL/df * df/dx_i = 2* f* 2* (x_i-i)
void  LossLayer::gradientCompute(const float avgLoss){
    //symbol deduced formula to compute gradient to prevLayerPoint->m_pdYVector
    // An Example
    long N = m_prevLayerPointer->m_pYVector->size();
    DynamicVector<float> & prevY = *(m_prevLayerPointer->m_pYVector);
    DynamicVector<float> & prevdY = *(m_prevLayerPointer->m_pdYVector);
    for (long i=0; i< N ;++i){
        prevdY[i] = 4*avgLoss* ( prevY[i] - i);
    }
}

float LossLayer::getLoss(){
    return m_loss;
}

float LossLayer::getAvgLoss(){
    return m_avgLoss;
}
float LossLayer::setAvgLoss(const float avgLoss){
    m_avgLoss = avgLoss;
}

