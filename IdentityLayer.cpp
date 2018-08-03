//
// Created by Sheen156 on 8/3/2018.
//

#include "IdentityLayer.h"

IdentityLayer::IdentityLayer(const int id, const string& name,Layer* prevLayer, Layer* nextLayer): Layer(id,name, {}){
    m_type = "IdentityLayer";
    addPreviousLayer(prevLayer);
    nextLayer->addPreviousLayer(this);
    m_pYTensor = prevLayer->m_pYTensor;
    m_pdYTensor = prevLayer->m_pdYTensor;
}

IdentityLayer::~IdentityLayer(){

}

// Y = X
// dL/dx = dL/dy * dy/dx = dL/dy
void IdentityLayer::forward(){
   //null
}
void IdentityLayer::backward(){
   //null;
}
void IdentityLayer::initialize(const string& initialMethod){
    //null
}

void IdentityLayer::zeroParaGradient(){
    //null
}

void IdentityLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}