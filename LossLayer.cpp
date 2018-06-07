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
}

LossLayer::~LossLayer(){

}

void LossLayer::forward(){
    lossCompute();
}
void LossLayer::backward(){
    gradientCompute();
}
void LossLayer::initialize(const string& initialMethod){
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

void  LossLayer::gradientCompute(){
    //symbol deduced formula to compute gradient to prevLayerPoint->m_pdYVector
    // An Example
    long N = m_prevLayerPointer->m_pYVector->size();
    DynamicVector<float> & prevY = *(m_prevLayerPointer->m_pYVector);
    DynamicVector<float> & prevdY = *(m_prevLayerPointer->m_pdYVector);
    for (long i=0; i< N ;++i){
        prevdY[i] = 2*( prevY[i] - i);
    }
}