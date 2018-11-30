//
// Created by Hui Xie on 11/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <SquareLoss.h>

#include "SquareLoss.h"

SquareLoss::SquareLoss(const int id, const string &name, Layer *prevLayer, float lambda) : LossLayer(id, name, prevLayer) {
    m_type = "SquareLoss";
    m_lambda = lambda;
}

SquareLoss::~SquareLoss() {
   //null;
}

void SquareLoss::printGroundTruth() {
   m_pGroundTruth->print();
}

// L= lambda*0.5*\sum (x_i- g_i)^2
float SquareLoss::lossCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const long N = X.getLength();
    float loss = 0;
    for (long i=0; i<N; ++i){
        loss += pow(X.e(i) - m_pGroundTruth->e(i), 2);
    }
    m_loss = loss*0.5*m_lambda;
    return m_loss;
 }

//dL/dx = (x_i-g_i)*lambda
void SquareLoss::gradientCompute() {
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    Tensor<float> & dX = *(m_prevLayer->m_pdYTensor);
    const long N = X.getLength();
    for (long i=0; i<N; ++i){
       dX[i] += (X.e(i) - m_pGroundTruth->e(i))*m_lambda;
    }
}

void SquareLoss::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, previousLayerIDs, outputTensorSize, filterSize, numFilter, FilterStride, startPosition, \r\n";
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, m_lambda, "{}");
}

void SquareLoss::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s: (%s, id=%d): PrevLayer=%s; Lambda=%f; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), m_lambda);
}
