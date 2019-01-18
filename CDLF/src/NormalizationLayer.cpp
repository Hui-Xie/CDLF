//
// Created by Hui Xie on 6/12/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include "NormalizationLayer.h"
#include "statisTool.h"
#include <math.h>       /* sqrt */
#include <NormalizationLayer.h>


NormalizationLayer::NormalizationLayer(const int id, const string& name,Layer* prevLayer, const vector<int>& tensorSize):Layer(id,name, tensorSize){
    m_type = "NormalizationLayer";
    addPreviousLayer(prevLayer);
    m_epsilon = 1e-6;
    m_sigma = m_epsilon;

    if (length(m_tensorSize) != length(m_prevLayer->m_tensorSize)){
        cout<<"Error: The output TensorSize does not equal with the one of the previous layer in Normalization construction at layID = "<<id<<endl;
    }
}
NormalizationLayer::~NormalizationLayer(){

}

void NormalizationLayer::initialize(const string& initialMethod){
     //null
}

void NormalizationLayer::zeroParaGradient(){
    //null
}

// Y = (X-mu)/sigma
// sigma = sigma + m_epsilon,   avoid sigma == 0
// dL/dX = dL/dY * dY/dX = dL/dY * (1/sigma)
void NormalizationLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    float mean = X.average();
    float m_sigma = sqrt(X.variance());
    m_sigma = (0 == m_sigma)? m_epsilon: m_sigma;
    const int N = Y.getLength();
    for (int i=0; i<N; ++i){
        Y.e(i) = (X.e(i) -mean)/m_sigma;
    }

    // below sentense will change Y's dims.discard
    //Y = (X-mean)/m_sigma;

}
void NormalizationLayer::backward(bool computeW, bool computeX){
    if(computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *(m_prevLayer->m_pdYTensor);
        const int N = dX.getLength();
        for (int i=0; i<N; ++i){
            dX.e(i) += dY.e(i)/m_sigma;
        }

        // below sentense will change Y's dims. discard
        // dX += dY/(m_sigma);
    }
}
void NormalizationLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}

int  NormalizationLayer::getNumParameters(){
    return 0;
}

void NormalizationLayer::save(const string &netDir) {
  //null;
}

void NormalizationLayer::load(const string &netDir) {
  //null;
}

void NormalizationLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, previousLayerIDs, outputTensorSize, filterSize, numFilter, FilterStride, startPosition, \r\n";
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void NormalizationLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}
