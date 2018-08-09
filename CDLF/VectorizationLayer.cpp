//
// Created by Hui Xie on 8/8/2018.
//

#include "VectorizationLayer.h"


VectorizationLayer::VectorizationLayer(const int id, const string& name,Layer* prevLayer)
   : Layer(id,name, {prevLayer->m_pYTensor->getLength(),1}){
    m_type = "VectorizationLayer";
    addPreviousLayer(prevLayer);
}

VectorizationLayer::~VectorizationLayer(){

}

void VectorizationLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    Y = X.vectorize();
}
void VectorizationLayer::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
    dX += dY.reshape(dX.getDims());
}
void VectorizationLayer::initialize(const string& initialMethod){
    //null
}

void VectorizationLayer::zeroParaGradient(){
    //null
}

void VectorizationLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}