//
// Created by Sheen156 on 6/6/2018.
//

#include "InputLayer.h"
#include "statisTool.h"

InputLayer::InputLayer(const int id, const string name, const long width): Layer(id, name, width){
    m_type = "InputLayer";

}

InputLayer::~InputLayer(){

}

void InputLayer::initialize(const string& initialMethod){
    // Gaussian random initialize
    if ("Gaussian" == initialMethod) {
        generateGaussian(m_pYTensor,0, 1 );
    }
}

void InputLayer::forward(){
    //null
}
void InputLayer::backward(){
    //null
}

void InputLayer::updateParameters(const float lr, const string& method){
    //null
}