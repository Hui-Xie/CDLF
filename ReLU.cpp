//
// Created by Sheen156 on 6/7/2018.
//

#include "ReLU.h"

ReLU::ReLU(Layer* preLayer): Layer(preLayer->m_width){
    m_type = "ReLU";
    m_prevLayerPointer = preLayer;
    m_prevLayerPointer->m_nextLayerPointer = this;


}

ReLU::~ReLU(){

}

// Y = X if X>=0; Y =0 else;
// dL/dx = dL/dy * dy/dx = dL/dy if X>=0; 0 if X < 0
void ReLU::forward(){
    DynamicVector<float>& Y = *m_pYVector;
    DynamicVector<float>& X = *m_prevLayerPointer->m_pYVector;
    for (long i=0; i< m_width; ++i){
       if (X[i] >= 0 ) Y[i] = X[i];
       else Y[i] = 0;
    }
}
void ReLU::backward(){
    DynamicVector<float>& dY = *m_pdYVector;
    DynamicVector<float>& dX = *m_prevLayerPointer->m_pdYVector;
    DynamicVector<float>& X = *m_prevLayerPointer->m_pYVector;
    for(long i=0; i< m_width; ++i){
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