//
// Created by Sheen156 on 6/6/2018.
//

#include "LossLayer.h"
#include <cmath>
#include <iostream>
using namespace std;

LossLayer::LossLayer(Layer* preLayer) : Layer(0){
    m_type = "LossLayer";
    m_prevLayerPointer = preLayer;
    m_prevLayerPointer->m_nextLayerPointer = this;
    m_loss = 1e+10;
}

LossLayer::~LossLayer(){

}
float LossLayer::getLoss(){
    return m_loss;
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

void LossLayer::updateParameters(const float lr, const string& method) {
    //null
}



