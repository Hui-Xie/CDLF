//
// Created by Sheen156 on 6/6/2018.
//

#include "LossLayer.h"
#include <cmath>
#include <iostream>
using namespace std;

LossLayer::LossLayer(const int id, const string& name) : Layer(id,name,{}){
    m_type = "LossLayer";
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

void LossLayer::zeroParaGradient(){
    //null
}

void LossLayer::updateParameters(const float lr, const string& method) {
    //null
}



