//
// Created by Sheen156 on 6/6/2018.
//

#include "InputLayer.h"
#include "statisTool.h"

InputLayer::InputLayer(const long width): Layer(width){
    m_type = "InputLayer";
    m_pYVector = new DynamicVector<float>(m_width);
    m_pdYVector = new DynamicVector<float>(m_width);
}

InputLayer::~InputLayer(){
    if (nullptr != m_pYVector) delete m_pYVector;
    if (nullptr != m_pdYVector) delete m_pdYVector;
}

void InputLayer::initialize(const string& initialMethod){
    // Gaussian random initialize
    if ("Gaussian" == initialMethod) {
        generateGaussian(m_pYVector,0, 1 );
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