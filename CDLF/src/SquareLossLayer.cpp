//
// Created by Hui Xie on 11/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <SquareLossLayer.h>

SquareLossLayer::SquareLossLayer(const int id, const string &name, Layer *prevLayer, float lambda) : LossLayer(id, name, prevLayer) {
    m_type = "SquareLossLayer";
    m_lambda = lambda;
}

SquareLossLayer::~SquareLossLayer() {
   //null;
}

void SquareLossLayer::printGroundTruth() {
    if (nullptr != m_pGroundTruth){
        m_pGroundTruth->print();
    }
}

// L= lambda*0.5*\sum (x_i- g_i)^2
float SquareLossLayer::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const long N = X.getLength();
    float loss = 0.0;
    for (long i=0; i<N; ++i){
        loss += pow(X.e(i) - m_pGroundTruth->e(i), 2);
    }
    m_loss = loss*0.5*m_lambda;
    return m_loss;
 }

//dL/dx = (x_i-g_i)*lambda
void SquareLossLayer::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    const long N = X.getLength();
    for (long i=0; i<N; ++i){
       dX[i] += (X.e(i) - m_pGroundTruth->e(i))*m_lambda;
    }
}

void SquareLossLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, previousLayerIDs, outputTensorSize, filterSize, numFilter, FilterStride, startPosition, \r\n";
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, m_lambda, "{}");
}

void SquareLossLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s, Type=%s, id=%d, PrevLayer=%s; Lambda=%f; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), m_lambda);
}
