//
// Created by Sheen156 on 6/7/2018.
//

#include "ReLU.h"

ReLU::ReLU(const int id, const string name,Layer* preLayer): Layer(id,name, preLayer->m_tensorSize){
    m_type = "ReLU";
    addPreviousLayer(preLayer);
}

ReLU::~ReLU(){

}

// Y = X if X>=0; Y =0 else;
// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
void ReLU::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayerPointer->m_pYTensor;
    for (long i=0; i< m_tensorSize; ++i){
       if (X[i] >= 0 ) Y[i] = X[i];
       else Y[i] = 0;
    }
}
void ReLU::backward(){
    Tensor<float>& dY = *m_pdYTensor;
    Tensor<float>& dX = *m_prevLayerPointer->m_pdYTensor;
    Tensor<float>& X = *m_prevLayerPointer->m_pYTensor;
    for(long i=0; i< m_tensorSize; ++i){
        if (X[i] >= 0) dX[i] = dY[i];
        else dX[i] = 0;
    }
}
void ReLU::initialize(const string& initialMethod){
    // this is null for ReLU
    return;
}

void ReLU::updateParameters(const float lr, const string& method) {
    //null
}