//
// Created by Sheen156 on 6/6/2018.
//

#include "InputLayer.h"

InputLayer::InputLayer(const long width): Layer(width){
    m_type = "Input";
    m_pYVector = new DynamicVector<float>(m_width);
    m_pdYVector = new DynamicVector<float>(m_width);
}

InputLayer::~InputLayer(){
    if (nullptr != m_pYVector) delete m_pYVector;
    if (nullptr != m_pdYVector) delete m_pdYVector;
}