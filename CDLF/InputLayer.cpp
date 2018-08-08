//
// Created by Hui Xie on 6/6/2018.
//

#include "InputLayer.h"
#include "statisTool.h"

InputLayer::InputLayer(const int id, const string& name, const vector<long>& tensorSize): Layer(id, name, tensorSize){
    m_type = "InputLayer";

}

InputLayer::~InputLayer(){

}

//this initialize method is just for random input case
void InputLayer::initialize(const string& initialMethod){
    //null
}

void InputLayer::zeroParaGradient(){
   //null
}

void InputLayer::forward(){
    //null
}
void InputLayer::backward(){
    //null
}

void InputLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}

void InputLayer::setInputTensor(const Tensor<float>& inputTensor){
    *m_pYTensor = inputTensor;
}